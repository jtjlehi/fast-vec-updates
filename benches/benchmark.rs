use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use fast_update::{Update, update_realloc, update_simple};
use rand::{Rng, SeedableRng, fill, rngs::StdRng};

fn build_input(length: usize) -> Vec<u8> {
    let mut v = vec![0; length];
    fill(&mut v[..]);
    v
}
fn build_updates(num_updates: usize, len: usize) -> Vec<Update> {
    let mut r = StdRng::from_os_rng();

    let mut v = (0..num_updates)
        .scan(len as i64, |length, _| {
            let index = r.random_range(0..len);
            Some(if r.random() {
                *length -= 1;
                Update::Remove(index)
            } else {
                *length += 1;
                Update::Insert(index, r.random())
            })
        })
        .collect::<Vec<_>>();
    v.sort();
    v
}
fn build_both(num_updates: usize, len: usize) -> (Vec<u8>, Vec<Update>) {
    (build_input(len), build_updates(num_updates, len))
}
pub fn bench_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("small");
    group.bench_function("update_simple", |b| {
        b.iter_batched(
            || build_both(5_000, 100_000),
            |(input, updates)| update_simple(&mut black_box(input), black_box(&updates)),
            criterion::BatchSize::SmallInput,
        )
    });
    group.bench_function("update_realloc", |b| {
        b.iter_batched(
            || build_both(5_000, 100_000),
            |(input, updates)| update_realloc(&mut black_box(input), black_box(&updates)),
            criterion::BatchSize::SmallInput,
        )
    });
    group.finish();
}
pub fn bench_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("large");
    group.sample_size(10);
    group.bench_function("update_sample", |b| {
        b.iter_batched(
            || build_both(50_000, 1_000_000),
            |(input, updates)| update_simple(&mut black_box(input), black_box(&updates)),
            criterion::BatchSize::SmallInput,
        )
    });
    group.bench_function("update_realloc", |b| {
        b.iter_batched(
            || build_both(50_000, 1_000_000),
            |(input, updates)| update_realloc(&mut black_box(input), black_box(&updates)),
            criterion::BatchSize::SmallInput,
        )
    });
    group.finish();
}
pub fn bench_extra_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("extra-large");
    group.bench_function("update_realloc", |b| {
        b.iter_batched(
            || build_both(5_000_000, 50_000_000),
            |(input, updates)| update_realloc(&mut black_box(input), black_box(&updates)),
            criterion::BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(benches, bench_small, bench_large, bench_extra_large);
criterion_main!(benches);
