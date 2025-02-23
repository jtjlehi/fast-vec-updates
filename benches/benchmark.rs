use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use fast_update::{build_both, update_collect_iter, update_realloc, update_simple, update_split};

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
            |(input, updates)| black_box(update_realloc(&black_box(input), black_box(&updates))),
            criterion::BatchSize::SmallInput,
        )
    });
    group.bench_function("update_collect_iter", |b| {
        b.iter_batched(
            || build_both(5_000, 100_000),
            |(input, updates)| {
                black_box(update_collect_iter(black_box(&input), black_box(&updates)))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_split", |b| {
        b.iter_batched(
            || build_both(5_000, 100_000),
            |(input, updates)| black_box(update_split(black_box(&input), black_box(&updates))),
            criterion::BatchSize::SmallInput,
        );
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
            |(input, updates)| black_box(update_realloc(&black_box(input), black_box(&updates))),
            criterion::BatchSize::SmallInput,
        )
    });
    group.bench_function("update_collect_iter", |b| {
        b.iter_batched(
            || build_both(50_000, 1_000_000),
            |(input, updates)| {
                black_box(update_collect_iter(black_box(&input), black_box(&updates)))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_split", |b| {
        b.iter_batched(
            || build_both(50_000, 1_000_000),
            |(input, updates)| black_box(update_split(black_box(&input), black_box(&updates))),
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}
pub fn bench_extra_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("extra-large");
    group.bench_function("update_realloc", |b| {
        b.iter_batched(
            || build_both(5_000_000, 50_000_000),
            |(input, updates)| black_box(update_realloc(&black_box(input), black_box(&updates))),
            criterion::BatchSize::SmallInput,
        )
    });
    group.bench_function("update_collect_iter", |b| {
        b.iter_batched(
            || build_both(5_000_000, 50_000_000),
            |(input, updates)| {
                black_box(update_collect_iter(black_box(&input), black_box(&updates)))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_split", |b| {
        b.iter_batched(
            || build_both(5_000_000, 50_000_000),
            |(input, updates)| black_box(update_split(black_box(&input), black_box(&updates))),
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_small, bench_large, bench_extra_large);
criterion_main!(benches);
