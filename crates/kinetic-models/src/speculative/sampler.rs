//! Picks the next token from logits.

use std::collections::HashMap;
use std::hash::{BuildHasher, RandomState};

use super::dry::{self, Breakers};
use super::filters::{self, Mirostat};
use super::{argmax, Logprobs};

/// How the next token is chosen.
#[derive(Clone, Debug, PartialEq)]
pub struct Sampling {
    /// Zero or below always takes the most likely token.
    pub temperature: f32,
    /// Keep only the `top_k` most likely tokens; zero keeps all.
    pub top_k: usize,
    /// Keep the most likely tokens whose probabilities add up to `top_p`.
    pub top_p: f32,
    /// Drop tokens less likely than `min_p` times the most likely one.
    pub min_p: f32,
    /// Divides positive logits of recently seen tokens and multiplies
    /// negative ones; 1 disables it.
    pub repeat_penalty: f32,
    /// Subtracted once from the logit of every recently seen token.
    pub presence_penalty: f32,
    /// Subtracted from the logit of a recent token for each time it appears.
    pub frequency_penalty: f32,
    /// How many recent tokens the penalties look at.
    pub repeat_last_n: usize,
    /// Makes sampled output repeatable; `None` picks a fresh seed.
    pub seed: Option<u64>,
    /// Added to the logits of these tokens before anything else.
    pub logit_bias: Vec<(u32, f32)>,
    /// Keeps the tokens closest to the expected surprise that add up to
    /// `typical_p`; 1 disables it.
    pub typical_p: f32,
    /// Keeps tokens within this many standard deviations of the highest
    /// logit; zero or below disables it.
    pub top_n_sigma: f32,
    /// Chance that XTC drops the likeliest tokens at a position.
    pub xtc_probability: f32,
    /// XTC drops every token at least this likely but the least likely one.
    pub xtc_threshold: f32,
    /// Strength of the DRY repetition penalty; zero disables it.
    pub dry_multiplier: f32,
    /// How fast the DRY penalty grows with the repetition's length.
    pub dry_base: f32,
    /// Repetitions this long go unpunished by DRY.
    pub dry_allowed_length: usize,
    /// How many recent tokens DRY looks at; `usize::MAX` for all.
    pub dry_penalty_last_n: usize,
    /// Texts that end a repetition for DRY.
    pub dry_sequence_breakers: Vec<String>,
    /// `dry_sequence_breakers` against the model's vocabulary, from
    /// [`dry_breakers`](super::dry_breakers); DRY sees no breakers without.
    pub dry_breaker_tokens: Vec<(u32, Vec<u32>)>,
    /// Mirostat version 1 or 2 in place of the penalties and every
    /// truncation, as in llama.cpp; 0 disables it.
    pub mirostat: u8,
    /// The surprise Mirostat aims at, in bits.
    pub mirostat_tau: f32,
    /// How fast Mirostat follows its target.
    pub mirostat_eta: f32,
    /// Widens `temperature` up to `temperature + dynatemp_range` on flat
    /// distributions; zero or below keeps the temperature fixed.
    pub dynatemp_range: f32,
    /// Curves how quickly the dynamic temperature reaches its widest; 1 is
    /// linear in the distribution's entropy.
    pub dynatemp_exponent: f32,
    /// Keeps at least this many candidates through every truncation, so a
    /// narrow `top_p` or `min_p` cannot leave a single token behind.
    pub min_keep: usize,
    /// Aims the choice at tokens this likely, away from both the head and
    /// the tail of the distribution; negative disables it.
    pub adaptive_target: f32,
    /// How much of the recent aiming survives each token; 0 forgets at
    /// once, close to 1 remembers for a long time.
    pub adaptive_decay: f32,
    /// Keeps generating after the model's end-of-sequence token, for prompt
    /// lengths and benchmarks that measure what the model would say next.
    pub ignore_eos: bool,
    /// Ends the turn once this many tokens were spent before the answer
    /// started, rather than letting reasoning run to the token limit.
    pub reasoning_budget: Option<usize>,
}

impl Default for Sampling {
    fn default() -> Self {
        Sampling {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_last_n: 64,
            seed: None,
            logit_bias: Vec::new(),
            typical_p: 1.0,
            top_n_sigma: -1.0,
            xtc_probability: 0.0,
            xtc_threshold: 0.1,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: usize::MAX,
            dry_sequence_breakers: ["\n", ":", "\"", "*"].map(String::from).to_vec(),
            dry_breaker_tokens: Vec::new(),
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            dynatemp_range: 0.0,
            dynatemp_exponent: 1.0,
            min_keep: 0,
            adaptive_target: -1.0,
            adaptive_decay: 0.9,
            ignore_eos: false,
            reasoning_budget: None,
        }
    }
}

impl Sampling {
    /// Whether the choice is plain argmax over the model's logits, which
    /// [`Greedy`](super::Greedy) makes without reading them.
    pub fn is_greedy(&self) -> bool {
        self.top() == Some(1)
    }

    /// How many of the highest logits of a row decide the choice, with
    /// ties going to the lower token; `None` when the whole row matters.
    /// Penalties and biases count as the whole row: covering every token
    /// they could let into the highest costs about as much as the full row
    /// on large vocabularies.
    pub fn top(&self) -> Option<usize> {
        if self.modifies_logits() {
            None
        } else if self.temperature <= 0.0 {
            Some(1)
        } else if self.mirostat != 0 || self.top_n_sigma > 0.0 {
            None
        } else {
            (self.top_k > 0).then_some(self.top_k)
        }
    }

    fn modifies_logits(&self) -> bool {
        !self.logit_bias.is_empty()
            || (self.mirostat == 0 && (self.has_penalties() || dry::is_active(self)))
    }

    fn has_penalties(&self) -> bool {
        self.repeat_last_n > 0
            && (self.repeat_penalty != 1.0
                || self.presence_penalty != 0.0
                || self.frequency_penalty != 0.0)
    }
}

/// Applies [`Sampling`] to one sequence's logits, token after token.
pub struct Sampler {
    params: Sampling,
    rng: SplitMix64,
    candidates: Vec<(u32, f32)>,
    /// The logits after biases and penalties.
    modified: Vec<f32>,
    breakers: Breakers,
    mirostat: Mirostat,
    /// Probabilities the adaptive aim compares against, one per candidate.
    originals: Vec<f32>,
    adaptive: Option<Adaptive>,
    log: Option<Logprobs>,
}

/// The running average of how likely the chosen tokens were, which the
/// adaptive aim pulls back towards its target.
struct Adaptive {
    weighted_sum: f32,
    total_weight: f32,
}

impl Adaptive {
    /// Starts out believing it already hits the target, as llama.cpp does.
    fn new(target: f32, decay: f32) -> Self {
        Adaptive {
            weighted_sum: target / (1.0 - decay),
            total_weight: 1.0 / (1.0 - decay),
        }
    }
}

/// How far from the aimed probability the weighting spreads.
const ADAPTIVE_WIDTH: f32 = 0.3;
/// The logit at the aimed probability, and how fast it falls away.
const ADAPTIVE_PEAK: f32 = 5.0;
const ADAPTIVE_SHARPNESS: f32 = 10.0;

impl Sampler {
    pub fn new(params: Sampling) -> Self {
        let seed = params
            .seed
            .unwrap_or_else(|| RandomState::new().hash_one(std::time::SystemTime::now()));
        let adaptive = (params.adaptive_target >= 0.0
            && params.adaptive_decay > 0.0
            && params.adaptive_decay < 1.0)
            .then(|| Adaptive::new(params.adaptive_target, params.adaptive_decay));
        Sampler {
            breakers: dry::breakers(&params),
            mirostat: Mirostat {
                version: params.mirostat,
                tau: params.mirostat_tau,
                eta: params.mirostat_eta,
                mu: 2.0 * params.mirostat_tau,
            },
            params,
            rng: SplitMix64(seed),
            candidates: Vec::new(),
            modified: Vec::new(),
            originals: Vec::new(),
            adaptive,
            log: None,
        }
    }

    /// Records the log-probabilities of every chosen token in `log`, from
    /// the logits as the model gave them.
    pub fn with_logprobs(mut self, log: Logprobs) -> Self {
        self.log = Some(log);
        self
    }

    /// See [`Sampling::top`]; the whole row while recording logprobs.
    pub fn top(&self) -> Option<usize> {
        self.params.top().filter(|_| self.log.is_none())
    }

    /// Chooses the token after `history`, the sequence so far, from the
    /// model's `logits`. Draws one random number unless the choice is
    /// greedy, and one more where XTC could apply.
    ///
    /// The order is llama.cpp's default: logit bias, penalties, DRY,
    /// top-n-sigma, top-k, typical, top-p, min-p, XTC, temperature; with
    /// Mirostat, logit bias, temperature, Mirostat.
    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> u32 {
        let token = self.choose(logits, history);
        if let Some(log) = &self.log {
            log.record(logits, token);
        }
        token
    }

    fn choose(&mut self, raw: &[f32], history: &[u32]) -> u32 {
        let p = &self.params;
        let logits = if p.modifies_logits() {
            let m = &mut self.modified;
            m.clear();
            m.extend_from_slice(raw);
            for &(id, bias) in &p.logit_bias {
                if let Some(l) = m.get_mut(id as usize) {
                    *l += bias;
                }
            }
            if p.mirostat == 0 {
                if p.has_penalties() {
                    let recent = &history[history.len().saturating_sub(p.repeat_last_n)..];
                    penalize(m, recent, p);
                }
                dry::penalize(m, history, p, &self.breakers);
            }
            &self.modified[..]
        } else {
            raw
        };
        if p.temperature <= 0.0 {
            return argmax(logits);
        }

        let c = &mut self.candidates;
        c.clear();
        c.extend(
            (0u32..)
                .zip(logits.iter().copied())
                .filter(|x| x.1 > f32::NEG_INFINITY),
        );
        if c.is_empty() {
            return argmax(logits);
        }
        if p.mirostat != 0 {
            let r = self.rng.next_f32();
            return self.mirostat.choose(c, p.temperature, logits.len(), r);
        }
        if p.top_n_sigma > 0.0 {
            filters::top_n_sigma(c, p.top_n_sigma);
        }
        if p.top_k > 0 && p.top_k < c.len() {
            c.select_nth_unstable_by(p.top_k - 1, by_logit);
            c.truncate(p.top_k);
        }
        c.sort_unstable_by(by_logit);

        softmax(c, 1.0);
        if p.typical_p < 1.0 {
            filters::typical(c, p.typical_p, p.min_keep);
        }
        if p.top_p < 1.0 {
            let mut sum = 0.0;
            let keep = c
                .iter()
                .position(|&(_, prob)| {
                    sum += prob;
                    sum >= p.top_p
                })
                .map_or(c.len(), |i| i + 1);
            c.truncate(keep.max(p.min_keep));
        }
        if p.min_p > 0.0 {
            let floor = c[0].1 * p.min_p;
            let keep = c.iter().take_while(|&&(_, prob)| prob >= floor).count();
            c.truncate(keep.max(1).max(p.min_keep));
        }
        let rng = &mut self.rng;
        filters::xtc(
            c,
            p.xtc_probability,
            p.xtc_threshold,
            p.min_keep,
            || rng.next_f32(),
        );

        for (id, prob) in c.iter_mut() {
            *prob = logits[*id as usize];
        }
        let temperature = dynamic_temperature(c, p);
        if temperature <= 0.0 {
            return argmax(logits);
        }
        softmax(c, temperature);
        if self.adaptive.is_some() {
            return self.aim();
        }
        c[draw(c, self.rng.next_f32())].0
    }

    /// Chooses a token whose probability lands near the adapted target:
    /// aims at `adaptive_target`, pulled back by how likely the recent
    /// choices actually were, then weights every candidate by how close it
    /// sits to that probability.
    fn aim(&mut self) -> u32 {
        let target = self.params.adaptive_target.clamp(0.0, 1.0);
        let (decay, originals, c) = (
            self.params.adaptive_decay,
            &mut self.originals,
            &mut self.candidates,
        );
        let state = self.adaptive.as_mut().expect("aim without an adaptive state");
        let recent = state.weighted_sum / state.total_weight;
        let aimed = (2.0 * target - recent).clamp(0.0, 1.0);
        originals.clear();
        originals.extend(c.iter().map(|x| x.1));
        let mut max = f32::NEG_INFINITY;
        for (_, prob) in c.iter_mut() {
            let dist = ((*prob - aimed) / ADAPTIVE_WIDTH).abs();
            *prob = ADAPTIVE_PEAK - ADAPTIVE_SHARPNESS * dist * dist / (1.0 + dist);
            max = max.max(*prob);
        }
        let mut sum = 0.0;
        for (_, prob) in c.iter_mut() {
            *prob = (*prob - max).exp();
            sum += *prob;
        }
        for (_, prob) in c.iter_mut() {
            *prob /= sum;
        }
        let i = draw(c, self.rng.next_f32());
        let prob = originals[i];
        state.weighted_sum = prob + decay * state.weighted_sum;
        state.total_weight = 1.0 + decay * state.total_weight;
        c[i].0
    }
}

/// The temperature this step samples at: `temperature` widened towards
/// `dynatemp_range` by how flat the candidates still are, `temperature`
/// itself when the range is off. `c` holds logits, highest first.
fn dynamic_temperature(c: &[(u32, f32)], p: &Sampling) -> f32 {
    if p.dynatemp_range <= 0.0 || c.len() < 2 {
        return p.temperature;
    }
    let max = c[0].1;
    let sum: f32 = c.iter().map(|x| (x.1 - max).exp()).sum();
    let entropy: f32 = c
        .iter()
        .map(|x| (x.1 - max).exp() / sum)
        .filter(|p| *p > 0.0)
        .map(|p| -p * p.ln())
        .sum();
    // the flattest distribution this many candidates can have
    let widest = (c.len() as f32).ln();
    let flat = (entropy / widest).clamp(0.0, 1.0);
    let low = (p.temperature - p.dynatemp_range).max(0.0);
    let high = p.temperature + p.dynatemp_range;
    low + (high - low) * flat.powf(p.dynatemp_exponent)
}

/// Highest logit first, ties to the lower token.
pub(super) fn by_logit(a: &(u32, f32), b: &(u32, f32)) -> std::cmp::Ordering {
    b.1.total_cmp(&a.1).then(a.0.cmp(&b.0))
}

/// The index of probabilities `c` that uniform `r` falls on.
pub(super) fn draw(c: &[(u32, f32)], mut r: f32) -> usize {
    for (i, &(_, prob)) in c.iter().enumerate() {
        if r < prob {
            return i;
        }
        r -= prob;
    }
    c.len() - 1
}

/// Scales the probabilities in `c` to add up to one.
pub(super) fn normalize(c: &mut [(u32, f32)]) {
    let sum: f32 = c.iter().map(|x| x.1).sum();
    c.iter_mut().for_each(|x| x.1 /= sum);
}

fn penalize(logits: &mut [f32], recent: &[u32], p: &Sampling) {
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for &t in recent {
        *counts.entry(t).or_default() += 1;
    }
    for (&t, &n) in &counts {
        let Some(l) = logits.get_mut(t as usize) else {
            continue;
        };
        *l = if *l > 0.0 {
            *l / p.repeat_penalty
        } else {
            *l * p.repeat_penalty
        };
        *l -= p.presence_penalty + p.frequency_penalty * n as f32;
    }
}

/// Replaces the logits in `c`, highest first, by their probabilities at
/// `temperature`.
pub(super) fn softmax(c: &mut [(u32, f32)], temperature: f32) {
    let max = c[0].1;
    let mut sum = 0.0;
    for (_, v) in c.iter_mut() {
        *v = ((*v - max) / temperature).exp();
        sum += *v;
    }
    c.iter_mut().for_each(|(_, v)| *v /= sum);
}

/// A small, fast generator; sampling needs no cryptographic strength.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampler(params: Sampling) -> Sampler {
        Sampler::new(Sampling {
            seed: Some(7),
            ..params
        })
    }

    fn counts(params: Sampling, logits: &[f32], n: usize) -> Vec<usize> {
        let mut s = sampler(params);
        let mut counts = vec![0; logits.len()];
        for _ in 0..n {
            counts[s.sample(logits, &[]) as usize] += 1;
        }
        counts
    }

    #[test]
    fn zero_temperature_takes_the_argmax() {
        let mut s = sampler(Sampling {
            temperature: 0.0,
            ..Sampling::default()
        });
        assert_eq!(s.sample(&[0.1, 2.0, 1.0], &[]), 1);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let c = counts(
            Sampling {
                temperature: 1.5,
                top_k: 1,
                top_p: 1.0,
                min_p: 0.0,
                ..Sampling::default()
            },
            &[1.0, 1.2, 0.5],
            200,
        );
        assert_eq!(c, [0, 200, 0]);
    }

    #[test]
    fn top_p_keeps_the_smallest_set_reaching_it() {
        // Probabilities about 0.66, 0.24, 0.09: top_p 0.8 keeps two.
        let c = counts(
            Sampling {
                temperature: 1.0,
                top_k: 0,
                top_p: 0.8,
                min_p: 0.0,
                ..Sampling::default()
            },
            &[2.0, 1.0, 0.0],
            2000,
        );
        assert_eq!(c[2], 0);
        assert!(c[1] > 400 && c[0] > 1200, "{c:?}");
    }

    #[test]
    fn min_p_drops_unlikely_tokens() {
        let c = counts(
            Sampling {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.2,
                ..Sampling::default()
            },
            &[3.0, 2.0, 0.0, 1.0],
            2000,
        );
        // exp(-3) and exp(-2) relative to the best are under 0.2.
        assert_eq!((c[2], c[3]), (0, 0));
        assert!(c[1] > 0);
    }

    #[test]
    fn sampling_follows_the_distribution() {
        let c = counts(
            Sampling {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                ..Sampling::default()
            },
            &[0.0, 2f32.ln()],
            30_000,
        );
        let ratio = c[1] as f64 / c[0] as f64;
        assert!((ratio - 2.0).abs() < 0.1, "{c:?}");
    }

    #[test]
    fn same_seed_same_tokens() {
        let params = Sampling {
            temperature: 1.0,
            top_k: 0,
            ..Sampling::default()
        };
        let logits: Vec<f32> = (0..50).map(|i| (i % 7) as f32 * 0.3).collect();
        let run = || {
            let mut s = sampler(params.clone());
            (0..20).map(|_| s.sample(&logits, &[])).collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn repeat_penalty_discourages_recent_tokens() {
        let mut s = sampler(Sampling {
            temperature: 0.0,
            repeat_penalty: 2.0,
            ..Sampling::default()
        });
        assert_eq!(s.sample(&[1.5, 1.0], &[0]), 1);
        assert_eq!(s.sample(&[1.5, 1.0], &[1]), 0);
    }

    #[test]
    fn frequency_penalty_counts_occurrences() {
        let mut logits = [1.0, 1.0, -1.0];
        penalize(
            &mut logits,
            &[0, 0, 1, 2],
            &Sampling {
                repeat_penalty: 2.0,
                presence_penalty: 0.5,
                frequency_penalty: 0.25,
                ..Sampling::default()
            },
        );
        assert_eq!(logits, [0.5 - 0.5 - 0.5, 0.5 - 0.5 - 0.25, -2.0 - 0.75]);
    }

    #[test]
    fn only_neutral_settings_are_greedy() {
        let greedy = Sampling {
            temperature: 0.0,
            ..Sampling::default()
        };
        assert!(greedy.is_greedy());
        assert!(!Sampling {
            repeat_penalty: 1.1,
            ..greedy.clone()
        }
        .is_greedy());
        assert!(!Sampling::default().is_greedy());
        assert!(Sampling {
            temperature: 1.0,
            top_k: 1,
            ..Sampling::default()
        }
        .is_greedy());
    }

    #[test]
    fn uniform_floats_stay_in_range() {
        let mut rng = SplitMix64(1);
        assert!((0..10_000)
            .map(|_| rng.next_f32())
            .all(|x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn only_the_top_logits_matter() {
        let logits: Vec<f32> = (0..500).map(|i| ((i * 37) % 101) as f32 * 0.05).collect();
        for params in [
            Sampling::default(),
            Sampling {
                temperature: 0.0,
                ..Sampling::default()
            },
        ] {
            let top = params.top().unwrap();
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
            let mut short = vec![f32::NEG_INFINITY; logits.len()];
            order[..top].iter().for_each(|&i| short[i] = logits[i]);
            let (mut full_sampler, mut short_sampler) = (sampler(params.clone()), sampler(params));
            for _ in 0..200 {
                assert_eq!(
                    full_sampler.sample(&logits, &[]),
                    short_sampler.sample(&short, &[])
                );
            }
        }
        let penalized = Sampling {
            repeat_penalty: 1.1,
            ..Sampling::default()
        };
        assert_eq!(penalized.top(), None);
    }

    #[test]
    fn min_keep_holds_the_truncations_open() {
        // top_p 0 keeps one token, min_keep 3 keeps three
        let c = counts(
            Sampling {
                temperature: 1.0,
                top_k: 0,
                top_p: 0.0,
                min_p: 0.0,
                min_keep: 3,
                ..Sampling::default()
            },
            &[3.0, 2.0, 1.0, 0.0],
            1000,
        );
        assert_eq!(c[3], 0);
        assert!(c[0] > 0 && c[1] > 0 && c[2] > 0, "{c:?}");
    }

    #[test]
    fn dynamic_temperature_follows_the_entropy() {
        let flat = Sampling {
            temperature: 0.5,
            dynatemp_range: 1.0,
            ..Sampling::default()
        };
        // four equally likely tokens: entropy is the widest it can be
        let even = [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)];
        assert!((dynamic_temperature(&even, &flat) - 1.5).abs() < 1e-5);
        // one token far above the rest: entropy is nearly zero
        let peak = [(0, 10.0), (1, 0.0), (2, 0.0), (3, 0.0)];
        assert!(dynamic_temperature(&peak, &flat) < 0.01);
        // the range off leaves the temperature alone
        let fixed = Sampling {
            temperature: 0.5,
            ..Sampling::default()
        };
        assert_eq!(dynamic_temperature(&even, &fixed), 0.5);
    }

    #[test]
    fn adaptive_aims_away_from_the_likeliest_token() {
        let logits = [4.0, 2.0, 0.0, -2.0];
        let plain = counts(
            Sampling {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                ..Sampling::default()
            },
            &logits,
            1000,
        );
        let aimed = counts(
            Sampling {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                adaptive_target: 0.5,
                adaptive_decay: 0.9,
                ..Sampling::default()
            },
            &logits,
            1000,
        );
        // plain sampling takes the head about 87% of the time; aiming at
        // 0.5 prefers the tokens nearest that probability instead
        assert!(aimed[0] < plain[0] * 3 / 4, "plain {plain:?} aimed {aimed:?}");
        assert!(aimed[1] > plain[1] * 2, "plain {plain:?} aimed {aimed:?}");
        assert_eq!(aimed.iter().sum::<usize>(), 1000);
    }
}
