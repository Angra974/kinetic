//! Exactness of the split path on a real model: with the first layers on the
//! host, batches, drafts, rollbacks, resets and state restores must leave
//! every logit and every token where plain decoding leaves them.
//!
//! KINETIC_TEST_GGUF=model.gguf cargo test --release -p kinetic-models --features vulkan --test vulkan -- --ignored

#![cfg(feature = "vulkan")]

use kinetic_gguf::GgufFile;
use kinetic_models::vulkan::VulkanModel;

const PROMPT: [u32; 12] = [1, 400, 1000, 23, 5000, 77, 901, 12, 3333, 600, 45, 2024];
const NEXT: [u32; 6] = [300, 3000, 30, 7, 777, 90];

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// The largest logit of each row of `logits`.
fn argmax(logits: &[f32], vocab: usize) -> Vec<u32> {
    logits
        .chunks_exact(vocab)
        .map(|row| {
            let mut best = 0;
            for (i, v) in row.iter().enumerate() {
                if *v > row[best] {
                    best = i;
                }
            }
            best as u32
        })
        .collect()
}

/// One test, so one model on the device: capturing the decode graph on one
/// thread forbids other threads' synchronous copies in the same context.
#[test]
#[ignore = "needs KINETIC_TEST_GGUF"]
fn the_host_split_changes_no_token_of_the_state_machine() {
    let path = std::env::var("KINETIC_TEST_GGUF").expect("set KINETIC_TEST_GGUF");
    let file = GgufFile::open(&path).unwrap();
    let model = kinetic_models::load(&file).unwrap();
    let layers = model.n_layers();
    // A quarter of the layers, their routed experts included, on the host: a
    // split deep enough that a wrong host layer moves real tokens.
    let front = layers / 4;
    let split = VulkanModel::load_on(&model, 64, Some(layers - front)).unwrap();
    assert_eq!(split.host_layers(), front);

    a_batch_matches_token_by_token(&split);
    a_draft_inside_a_checkpoint_gives_plain_tokens(&split);
    a_rejected_draft_leaves_the_state_as_if_never_run(&split);
    rollback_leaves_the_state_as_if_never_run(&split);
    a_reset_state_is_a_fresh_one(&split);
    a_host_copy_restores_the_state(&split, &model);
    a_reused_state_restores_like_a_fresh_one(&split, &model);
    greedy_is_the_largest_logit(&split);
    the_cpu_path_predicts_the_same_tokens(&split, &model);
}

fn a_batch_matches_token_by_token(gpu: &VulkanModel) {
    let vocab = gpu.n_vocab();
    let mut batched = gpu.new_state().unwrap();
    gpu.run(&mut batched, &PROMPT, false).unwrap();
    let all = gpu.run(&mut batched, &NEXT, true).unwrap().logits;

    let mut single = gpu.new_state().unwrap();
    gpu.run(&mut single, &PROMPT, false).unwrap();
    for (i, &t) in NEXT.iter().enumerate() {
        let logits = gpu.run(&mut single, &[t], false).unwrap().logits;
        assert_eq!(bits(&logits), bits(&all[i * vocab..][..vocab]), "row {i}");
    }
}

/// A draft verified in one pass past a checkpoint, of which the first two
/// tokens are kept, gives the state, and so the tokens, plain decoding gives:
/// the loop a speculative session runs on the host split.
fn a_draft_inside_a_checkpoint_gives_plain_tokens(gpu: &VulkanModel) {
    let mut drafted = gpu.new_state().unwrap();
    gpu.run(&mut drafted, &PROMPT, false).unwrap();
    gpu.checkpoint(&mut drafted).unwrap();
    let (candidates, _) = gpu.run_greedy(&mut drafted, &NEXT, true).unwrap();
    assert_eq!(candidates.len(), NEXT.len());
    gpu.rollback(&mut drafted, 2).unwrap();
    let after = gpu
        .run(&mut drafted, &[candidates[2]], false)
        .unwrap()
        .logits;

    let mut plain = gpu.new_state().unwrap();
    gpu.run(&mut plain, &PROMPT, false).unwrap();
    gpu.run(&mut plain, &NEXT[..2], false).unwrap();
    let expected = gpu.run(&mut plain, &[candidates[2]], false).unwrap().logits;
    assert_eq!(drafted.len(), plain.len());
    assert_eq!(bits(&after), bits(&expected));
}

/// Every drafted token rejected: the state goes back to the checkpoint, host
/// layers included.
fn a_rejected_draft_leaves_the_state_as_if_never_run(gpu: &VulkanModel) {
    let mut rejected = gpu.new_state().unwrap();
    gpu.run(&mut rejected, &PROMPT, false).unwrap();
    gpu.checkpoint(&mut rejected).unwrap();
    gpu.run(&mut rejected, &NEXT, true).unwrap();
    gpu.rollback(&mut rejected, 0).unwrap();
    let after = gpu.run(&mut rejected, &[42, 43], true).unwrap().logits;

    let mut plain = gpu.new_state().unwrap();
    gpu.run(&mut plain, &PROMPT, false).unwrap();
    let expected = gpu.run(&mut plain, &[42, 43], true).unwrap().logits;
    assert_eq!(rejected.len(), plain.len());
    assert_eq!(bits(&after), bits(&expected));
}

fn rollback_leaves_the_state_as_if_never_run(gpu: &VulkanModel) {
    let mut rolled = gpu.new_state().unwrap();
    gpu.run(&mut rolled, &PROMPT, false).unwrap();
    gpu.checkpoint(&mut rolled).unwrap();
    gpu.run(&mut rolled, &NEXT, true).unwrap();
    gpu.rollback(&mut rolled, 2).unwrap();
    let after_rollback = gpu.run(&mut rolled, &[42], false).unwrap().logits;

    let mut plain = gpu.new_state().unwrap();
    gpu.run(&mut plain, &PROMPT, false).unwrap();
    gpu.run(&mut plain, &NEXT[..2], false).unwrap();
    let expected = gpu.run(&mut plain, &[42], false).unwrap().logits;
    assert_eq!(rolled.len(), plain.len());
    assert_eq!(bits(&after_rollback), bits(&expected));
}

/// A state reset after a sequence gives what a new one gives, on every layer,
/// those on the host too.
fn a_reset_state_is_a_fresh_one(gpu: &VulkanModel) {
    let mut fresh = gpu.new_state().unwrap();
    let expected = gpu.run(&mut fresh, &NEXT, true).unwrap().logits;

    let mut reused = gpu.new_state().unwrap();
    gpu.run(&mut reused, &PROMPT, false).unwrap();
    gpu.reset(&mut reused).unwrap();
    assert!(reused.is_empty());
    let logits = gpu.run(&mut reused, &NEXT, true).unwrap().logits;
    assert_eq!(bits(&logits), bits(&expected));
}

/// The warm cache stores a host copy and brings it back on another state.
fn a_host_copy_restores_the_state(gpu: &VulkanModel, model: &kinetic_models::Model) {
    let mut state = gpu.new_state().unwrap();
    gpu.run(&mut state, &PROMPT, false).unwrap();
    let host = gpu.host_state(model, &state).unwrap();
    let mut restored = gpu.device_state(&host).unwrap();
    let expected = gpu.run(&mut state, &NEXT, true).unwrap().logits;
    let logits = gpu.run(&mut restored, &NEXT, true).unwrap().logits;
    assert_eq!(bits(&logits), bits(&expected));
}

/// The warm cache reuses a spare state instead of allocating one; loading
/// into a state that already ran something must give the same as a fresh
/// state.
fn a_reused_state_restores_like_a_fresh_one(gpu: &VulkanModel, model: &kinetic_models::Model) {
    let mut source = gpu.new_state().unwrap();
    gpu.run(&mut source, &PROMPT, false).unwrap();
    let host = gpu.host_state(model, &source).unwrap();
    let expected = gpu.run(&mut source, &NEXT, true).unwrap().logits;

    let mut spare = gpu.new_state().unwrap();
    gpu.run(&mut spare, &NEXT, true).unwrap();
    gpu.load_state(&mut spare, &host).unwrap();
    let logits = gpu.run(&mut spare, &NEXT, true).unwrap().logits;
    assert_eq!(bits(&logits), bits(&expected));
}

/// The greedy tokens a state asks for are the largest logits it would give
/// out, rows shared by the batch included.
fn greedy_is_the_largest_logit(gpu: &VulkanModel) {
    let vocab = gpu.n_vocab();
    let mut state = gpu.new_state().unwrap();
    gpu.run(&mut state, &PROMPT, false).unwrap();
    let (greedy, _) = gpu.run_greedy(&mut state, &NEXT, true).unwrap();

    let mut other = gpu.new_state().unwrap();
    gpu.run(&mut other, &PROMPT, false).unwrap();
    let all = gpu.run(&mut other, &NEXT, true).unwrap().logits;
    assert_eq!(greedy, argmax(&all, vocab));
}

/// The layers on the host compute what the CPU path computes: the tokens of
/// a short continuation are the same, so a wrong host layer cannot pass for
/// a rounding difference.
fn the_cpu_path_predicts_the_same_tokens(gpu: &VulkanModel, model: &kinetic_models::Model) {
    let vocab = gpu.n_vocab();
    let mut cpu = model.new_state();
    let cpu_logits = model.run(&mut cpu, &PROMPT, false).logits;
    let mut dev = gpu.new_state().unwrap();
    let dev_logits = gpu.run(&mut dev, &PROMPT, false).unwrap().logits;

    let gap = cpu_logits
        .iter()
        .zip(&dev_logits)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    // Qwen3.5-4B Q4_K_M measures 0.58 of ~10 between the CPU and the whole
    // model on the device; a host layer that is wrong, not rounded, is far
    // above that.
    assert!(gap < 3.0, "{gap} between the CPU path and the host split");

    let mut token = argmax(&dev_logits, vocab)[0];
    for step in 0..8 {
        let cpu_next = model.run(&mut cpu, &[token], false).logits;
        let dev_next = gpu.run(&mut dev, &[token], false).unwrap().logits;
        assert_eq!(
            argmax(&cpu_next, vocab),
            argmax(&dev_next, vocab),
            "step {step}"
        );
        token = argmax(&dev_next, vocab)[0];
    }
}
