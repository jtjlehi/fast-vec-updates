use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use fast_update::{
    build_both, contig_updates::ContigUpdates, init_new_types_1, update_collect_iter,
    update_in_chunks, update_new_types_1_pre_compute, update_new_types_3_pre_compute,
    update_realloc, update_simple, update_split_alloc, update_split_new_types,
    update_split_new_types_1, update_split_new_types_1_1, update_split_new_types_2,
    update_split_new_types_3,
};

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
    group.bench_function("update_split_alloc", |b| {
        b.iter_batched(
            || build_both(5_000, 100_000),
            |(input, updates)| {
                black_box(update_split_alloc(black_box(&input), black_box(&updates)))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_split_new_types", |b| {
        b.iter_batched(
            || build_both(5_000, 100_000),
            |(input, updates)| {
                black_box(update_split_new_types(
                    black_box(&input),
                    black_box(&updates),
                ))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_split_new_types_1", |b| {
        b.iter_batched(
            || build_both(5_000, 100_000),
            |(input, updates)| {
                black_box(update_split_new_types_1(
                    black_box(&input),
                    black_box(&updates),
                ))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_split_new_types_1_1", |b| {
        b.iter_batched(
            || build_both(5_000, 100_000),
            |(input, updates)| {
                black_box(update_split_new_types_1_1(
                    black_box(&input),
                    black_box(&updates),
                ))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_split_new_types_2", |b| {
        b.iter_batched(
            || build_both(5_000, 100_000),
            |(input, updates)| {
                black_box(update_split_new_types_2(
                    black_box(&input),
                    black_box(&updates),
                ))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_split_new_types_3", |b| {
        b.iter_batched(
            || build_both(5_000, 100_000),
            |(input, updates)| {
                black_box(update_split_new_types_3(
                    black_box(&input),
                    black_box(&updates),
                ))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_in_chunks", |b| {
        b.iter_batched(
            || build_both(5_000, 100_000),
            |(input, updates)| black_box(update_in_chunks(black_box(&input), black_box(&updates))),
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}
pub fn bench_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("large");
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
    group.bench_function("update_split_alloc", |b| {
        b.iter_batched(
            || build_both(50_000, 1_000_000),
            |(input, updates)| {
                black_box(update_split_alloc(black_box(&input), black_box(&updates)))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_split_new_types", |b| {
        b.iter_batched(
            || build_both(50_000, 1_000_000),
            |(input, updates)| {
                black_box(update_split_new_types(
                    black_box(&input),
                    black_box(&updates),
                ))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_split_new_types_1", |b| {
        b.iter_batched(
            || build_both(50_000, 1_000_000),
            |(input, updates)| {
                black_box(update_split_new_types_1(
                    black_box(&input),
                    black_box(&updates),
                ))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_split_new_types_1_1", |b| {
        b.iter_batched(
            || build_both(50_000, 1_000_000),
            |(input, updates)| {
                black_box(update_split_new_types_1_1(
                    black_box(&input),
                    black_box(&updates),
                ))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_split_new_types_2", |b| {
        b.iter_batched(
            || build_both(50_000, 1_000_000),
            |(input, updates)| {
                black_box(update_split_new_types_2(
                    black_box(&input),
                    black_box(&updates),
                ))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_split_new_types_3", |b| {
        b.iter_batched(
            || build_both(50_000, 1_000_000),
            |(input, updates)| {
                black_box(update_split_new_types_3(
                    black_box(&input),
                    black_box(&updates),
                ))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

pub fn bench_precomputed(c: &mut Criterion) {
    let mut group = c.benchmark_group("precomputed");
    group.bench_function("new_types_1", |b| {
        b.iter_batched(
            || {
                let (input, updates) = build_both(50_000, 1_000_000);

                let (removes, inserts) = init_new_types_1(&updates);
                let inserts_vec = Vec::with_capacity(input.len() + inserts.len());

                let output = Vec::with_capacity(input.len() + inserts.len() - removes.len());
                (input, removes, inserts, inserts_vec, output)
            },
            |(input, removes, inserts, mut inserts_vec, mut output)| {
                update_new_types_1_pre_compute(
                    black_box(&input),
                    black_box(&removes),
                    black_box(&inserts),
                    black_box(&mut inserts_vec),
                    black_box(&mut output),
                );
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("new_types_3", |b| {
        b.iter_batched(
            || {
                let (input, updates) = build_both(50_000, 1_000_000);

                let contig_updates = ContigUpdates::new(&updates);

                let inserts_vec = contig_updates.alloc_inserts_vec(input.len());
                let output = contig_updates.alloc_removes_vec(inserts_vec.capacity());
                (input, contig_updates, inserts_vec, output)
            },
            |(input, contig_updates, mut inserts_vec, mut output)| {
                update_new_types_3_pre_compute(
                    black_box(&input),
                    black_box(&contig_updates),
                    black_box(&mut inserts_vec),
                    black_box(&mut output),
                );
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

pub fn bench_large_new_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_new_types");
    group.bench_function("update_split_new_types_1", |b| {
        b.iter_batched(
            || build_both(50_000, 1_000_000),
            |(input, updates)| {
                black_box(update_split_new_types_1(
                    black_box(&input),
                    black_box(&updates),
                ))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_split_new_types_1_1", |b| {
        b.iter_batched(
            || build_both(50_000, 1_000_000),
            |(input, updates)| {
                black_box(update_split_new_types_1_1(
                    black_box(&input),
                    black_box(&updates),
                ))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_split_new_types_2", |b| {
        b.iter_batched(
            || build_both(50_000, 1_000_000),
            |(input, updates)| {
                black_box(update_split_new_types_2(
                    black_box(&input),
                    black_box(&updates),
                ))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_split_new_types_3", |b| {
        b.iter_batched(
            || build_both(50_000, 1_000_000),
            |(input, updates)| {
                black_box(update_split_new_types_3(
                    black_box(&input),
                    black_box(&updates),
                ))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}
pub fn bench_extra_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("extra-large");
    group.sample_size(10);
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
    group.bench_function("update_split_alloc", |b| {
        b.iter_batched(
            || build_both(5_000_000, 50_000_000),
            |(input, updates)| {
                black_box(update_split_alloc(black_box(&input), black_box(&updates)))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_split_new_types", |b| {
        b.iter_batched(
            || build_both(5_000_000, 50_000_000),
            |(input, updates)| {
                black_box(update_split_new_types(
                    black_box(&input),
                    black_box(&updates),
                ))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_split_new_types_1", |b| {
        b.iter_batched(
            || build_both(5_000_000, 50_000_000),
            |(input, updates)| {
                black_box(update_split_new_types_1(
                    black_box(&input),
                    black_box(&updates),
                ))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_split_new_types_1_1", |b| {
        b.iter_batched(
            || build_both(5_000_000, 50_000_000),
            |(input, updates)| {
                black_box(update_split_new_types_1_1(
                    black_box(&input),
                    black_box(&updates),
                ))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_split_new_types_2", |b| {
        b.iter_batched(
            || build_both(5_000_000, 50_000_000),
            |(input, updates)| {
                black_box(update_split_new_types_2(
                    black_box(&input),
                    black_box(&updates),
                ))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("update_split_new_types_3", |b| {
        b.iter_batched(
            || build_both(5_000_000, 50_000_000),
            |(input, updates)| {
                black_box(update_split_new_types_3(
                    black_box(&input),
                    black_box(&updates),
                ))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_small,
    bench_large,
    bench_large_new_types,
    bench_extra_large,
    bench_precomputed,
);
criterion_main!(benches);
