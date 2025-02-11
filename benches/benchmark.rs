use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use fast_update::add;

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("add 20", |b| b.iter(|| add(black_box(20), black_box(20))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
