#![feature(let_chains)]
#![feature(maybe_uninit_as_bytes)]
#![feature(maybe_uninit_slice)]
//! all of the functions update the inputs based on the slice of updates

/// module to encapsulate `ContigUpdates`
pub mod contig_updates;

use core::mem::MaybeUninit;

use contig_updates::ContigUpdates;

pub fn build_input(length: usize) -> Vec<u8> {
    let mut v = vec![0; length];
    rand::fill(&mut v[..]);
    v
}
pub fn build_updates(num_updates: usize, len: usize) -> Vec<Update> {
    use rand::{Rng, SeedableRng, rngs::StdRng};
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
pub fn build_both(num_updates: usize, len: usize) -> (Vec<u8>, Vec<Update>) {
    (build_input(len), build_updates(num_updates, len))
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Update {
    Remove(usize),
    Insert(usize, u8),
}

impl Update {
    #[inline]
    fn index(self) -> usize {
        match self {
            Update::Remove(idx) => idx,
            Update::Insert(idx, _) => idx,
        }
    }
    #[inline]
    fn offset_index(self, offset: i64) -> usize {
        let idx = self.index() as i64 + offset;
        if idx < 0 { 0 } else { idx as usize }
    }
}

impl Ord for Update {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.index().cmp(&other.index()) {
            std::cmp::Ordering::Equal => match (self, other) {
                (Update::Remove(_), Update::Remove(_))
                | (Update::Insert(_, _), Update::Insert(_, _)) => std::cmp::Ordering::Equal,
                (Update::Remove(_), Update::Insert(_, _)) => std::cmp::Ordering::Greater,
                (Update::Insert(_, _), Update::Remove(_)) => std::cmp::Ordering::Less,
            },
            ordering => ordering,
        }
    }
}
impl PartialOrd for Update {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// basic for loop implementation: go through all updates and apply them
///
/// (removals are idempotent)
///
/// used as a baseline test for all other implementations
pub fn update_simple(input: &mut Vec<u8>, updates: &[Update]) {
    let mut offset: i64 = 0;
    let mut last_index = None;
    for update in updates {
        let index = update.offset_index(offset);
        match *update {
            // idempotent removal
            Update::Remove(idx) if Some(idx) == last_index => continue,
            Update::Remove(idx) => {
                input.remove(index);
                last_index = Some(idx);
                offset -= 1;
            }
            Update::Insert(_, value) => {
                input.insert(index, value);
                offset += 1;
            }
        }
    }
}

struct UpdateIter<'updates, 'inputs> {
    updates: &'updates [Update],
    update_idx: usize,
    input: &'inputs [u8],
    input_idx: usize,
}

impl Iterator for UpdateIter<'_, '_> {
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        assert!(
            self.updates
                .get(self.update_idx)
                .map(|update| update.index() >= self.input_idx)
                .unwrap_or(true),
            "skipped an update, (input_idx = {}, update_idx = {}, and update = {:?})",
            self.input_idx,
            self.update_idx,
            self.updates.get(self.update_idx)
        );

        // skip through the remove updates
        loop {
            match self.updates.get(self.update_idx) {
                Some(Update::Remove(idx)) if self.input_idx == *idx => {
                    self.update_idx += 1;
                    self.input_idx += 1;
                }
                Some(Update::Remove(idx)) if self.input_idx > *idx => {
                    // idempotent removal
                    self.update_idx += 1;
                }
                _ => break,
            }
        }

        match self.updates.get(self.update_idx) {
            Some(Update::Remove(idx)) if self.input_idx == *idx => unreachable!(),
            Some(Update::Insert(idx, val)) if self.input_idx >= *idx => {
                self.update_idx += 1;
                Some(*val)
            }
            _ => {
                let nxt = self.input.get(self.input_idx).cloned();
                self.input_idx += 1;
                nxt
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.input.len()
            + self
                .updates
                .iter()
                .filter(|update| matches!(update, Update::Insert(_, _)))
                .count()
            - self
                .updates
                .iter()
                .filter(|update| matches!(update, Update::Remove(_)))
                .count();
        (size, Some(size))
    }
}

pub fn update_collect_iter(input: &[u8], updates: &[Update]) -> Vec<u8> {
    UpdateIter {
        updates,
        update_idx: 0,
        input,
        input_idx: 0,
    }
    .collect()
}

/// allocates a new array and inserts elements from vector and array
///
/// calculates exact size of iterator before hand
/// (uses iterators)
pub fn update_realloc(input: &[u8], updates: &[Update]) -> Vec<u8> {
    // TODO: fix this one
    // calculate the change in the vector size
    let total_offset: i64 = updates
        .iter()
        .map(|update| match update {
            Update::Remove(_) => -1,
            Update::Insert(_, _) => 1,
        })
        .sum();
    let mut v = Vec::with_capacity((input.len() as i64 + total_offset) as usize);

    let mut updates = updates.iter();
    let mut inputs = input.iter().enumerate();

    let mut update = updates.next();
    let mut next_input = inputs.next();

    // loop until no updates apply to current idx
    loop {
        match (update, next_input) {
            (Some(Update::Remove(index)), Some((idx, _))) if *index == idx => {
                update = updates.next();
                next_input = inputs.next();
            }
            (Some(Update::Remove(index)), Some((idx, _))) if *index < idx => {
                update = updates.next();
            }
            (Some(Update::Insert(index, insert_val)), Some((idx, _))) if *index == idx => {
                v.push(*insert_val);
                update = updates.next();
            }
            (_, Some((_, val))) => {
                v.push(*val);
                next_input = inputs.next();
            }
            (None, None) => break,
            (Some(Update::Remove(_)), None) => {
                update = updates.next();
            }
            (Some(Update::Insert(_, insert_val)), None) => {
                v.push(*insert_val);
                update = updates.next();
            }
        }
    }

    v
}

#[inline]
fn split_alloc(updates: &[Update]) -> (Vec<Update>, Vec<Update>) {
    updates
        .iter()
        .copied()
        .scan(0, |num_inserts, update| {
            Some(match update {
                Update::Remove(idx) => Update::Remove(idx + *num_inserts),
                insert @ Update::Insert(_, _) => {
                    *num_inserts += 1;
                    insert
                }
            })
        })
        .partition(|updates| matches!(updates, Update::Remove(_)))
}

#[inline]
fn split_insert(input: &[u8], inserts: Vec<Update>) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len() + inserts.len());
    let mut inserts = inserts.into_iter();
    let mut next_insert = inserts.next();
    for (idx, &val) in input.iter().enumerate() {
        while let Some(Update::Insert(insert_idx, insert_val)) = next_insert {
            if insert_idx == idx {
                next_insert = inserts.next();
                output.push(insert_val);
            } else {
                break;
            }
        }
        // we always push the current val
        output.push(val);
    }
    // continue to get the items out
    let curr_idx = output.len();
    while let Some(Update::Insert(insert_idx, insert_val)) = next_insert {
        if insert_idx == curr_idx {
            output.push(insert_val);
            next_insert = inserts.next();
        } else {
            break;
        }
    }

    output
}

#[inline]
fn split_remove(input: &[u8], removes: Vec<Update>) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len() - removes.len());
    let mut removes = removes.into_iter();
    let mut next_remove = removes.next();
    for (idx, val) in input.iter().enumerate() {
        let mut add_val = true;
        while let Some(Update::Remove(remove_idx)) = next_remove {
            if remove_idx == idx {
                next_remove = removes.next();
                add_val = false;
            } else {
                break;
            }
        }
        if add_val {
            output.push(*val)
        }
    }
    output
}

pub fn update_split_alloc(input: &[u8], updates: &[Update]) -> Vec<u8> {
    let (removes, inserts): (Vec<_>, Vec<_>) = split_alloc(updates);

    split_remove(&split_insert(input, inserts), removes)
}

#[inline]
fn split_new_types(updates: &[Update]) -> (Vec<usize>, Vec<(usize, u8)>) {
    let mut removes = Vec::<usize>::with_capacity(updates.len());
    let mut inserts = Vec::<(usize, u8)>::with_capacity(updates.len());

    for update in updates {
        match *update {
            Update::Remove(idx) => removes.push(idx + inserts.len()),
            Update::Insert(idx, val) => inserts.push((idx, val)),
        }
    }
    (removes, inserts)
}
#[inline]
fn insert_val_indexes(input: &[u8], inserts: &[(usize, u8)]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len() + inserts.len());
    let mut inserts = inserts.iter();
    let mut next_insert = inserts.next();
    for (idx, &val) in input.iter().enumerate() {
        while let Some(&(insert_idx, insert_val)) = next_insert {
            if insert_idx == idx {
                next_insert = inserts.next();
                output.push(insert_val);
            } else {
                break;
            }
        }
        // we always push the current val
        output.push(val);
    }
    // continue to get the items out
    let curr_idx = output.len();
    while let Some(&(insert_idx, insert_val)) = next_insert {
        if insert_idx == curr_idx {
            output.push(insert_val);
            next_insert = inserts.next();
        } else {
            break;
        }
    }

    output
}

#[inline]
fn remove_indexes(input: &[u8], removes: &[usize]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len() - removes.len());
    let mut prev_idx = 0;
    for remove_idx in removes {
        if remove_idx < &prev_idx {
            continue;
        }
        output.extend_from_slice(&input[prev_idx..*remove_idx]);
        prev_idx = remove_idx + 1;
    }
    output.extend_from_slice(&input[prev_idx..]);
    output
}

pub fn update_split_new_types(input: &[u8], updates: &[Update]) -> Vec<u8> {
    let (removes, inserts) = split_new_types(updates);

    remove_indexes(&insert_val_indexes(input, &inserts), &removes)
}

pub fn update_split_new_types_1(input: &[u8], updates: &[Update]) -> Vec<u8> {
    let (removes, inserts) = split_new_types(updates);

    let mut inserted = Vec::with_capacity(input.len() + inserts.len());
    let mut prev_idx = 0;
    for (insert_idx, val) in inserts {
        inserted.extend_from_slice(&input[prev_idx..insert_idx]);
        inserted.push(val);
        prev_idx = insert_idx;
    }
    inserted.extend_from_slice(&input[prev_idx..]);

    let mut output = Vec::with_capacity(inserted.len() - removes.len());
    let mut prev_idx = 0;
    for remove_idx in removes {
        if remove_idx < prev_idx {
            continue;
        }
        output.extend_from_slice(&inserted[prev_idx..remove_idx]);
        prev_idx = remove_idx + 1;
    }
    output.extend_from_slice(&inserted[prev_idx..]);
    output
}
pub fn update_split_new_types_1_1(input: &[u8], updates: &[Update]) -> Vec<u8> {
    let (removes, inserts) = {
        let mut removes = Vec::<usize>::with_capacity(updates.len());
        let mut inserts = Vec::<(usize, u8)>::with_capacity(updates.len());

        for update in updates {
            match *update {
                Update::Remove(idx) => {
                    let remove_idx = idx + inserts.len();
                    if Some(&remove_idx) != removes.last() {
                        removes.push(idx + inserts.len())
                    }
                }
                Update::Insert(idx, val) => inserts.push((idx, val)),
            }
        }
        (removes, inserts)
    };

    let mut inserted = Vec::with_capacity(input.len() + inserts.len());
    let mut prev_idx = 0;
    for (insert_idx, val) in inserts {
        inserted.extend_from_slice(&input[prev_idx..insert_idx]);
        inserted.push(val);
        prev_idx = insert_idx;
    }
    inserted.extend_from_slice(&input[prev_idx..]);

    let mut output = Vec::with_capacity(inserted.len() - removes.len());
    let mut prev_idx = 0;
    for remove_idx in removes {
        output.extend_from_slice(&inserted[prev_idx..remove_idx]);
        prev_idx = remove_idx + 1;
    }
    output.extend_from_slice(&inserted[prev_idx..]);
    output
}
pub fn init_new_types_1(updates: &[Update]) -> (Vec<usize>, Vec<(usize, u8)>) {
    let mut removes = Vec::<usize>::with_capacity(updates.len());
    let mut inserts = Vec::<(usize, u8)>::with_capacity(updates.len());

    for update in updates {
        match *update {
            Update::Remove(idx) => {
                let remove_idx = idx + inserts.len();
                if Some(&remove_idx) != removes.last() {
                    removes.push(idx + inserts.len())
                }
            }
            Update::Insert(idx, val) => inserts.push((idx, val)),
        }
    }
    (removes, inserts)
}
pub fn update_new_types_1_pre_compute(
    input: &[u8],
    removes: &[usize],
    inserts: &[(usize, u8)],
    inserted: &mut Vec<u8>,
    output: &mut Vec<u8>,
) {
    assert!(inserted.capacity() >= input.len() + inserts.len());
    assert!(output.capacity() >= input.len() - removes.len() + inserts.len());

    let mut prev_idx = 0;
    for &(insert_idx, val) in inserts {
        inserted.extend_from_slice(&input[prev_idx..insert_idx]);
        inserted.push(val);
        prev_idx = insert_idx;
    }
    inserted.extend_from_slice(&input[prev_idx..]);

    let mut prev_idx = 0;
    for &remove_idx in removes {
        output.extend_from_slice(&inserted[prev_idx..remove_idx]);
        prev_idx = remove_idx + 1;
    }
    output.extend_from_slice(&inserted[prev_idx..]);
}

struct PartialInitSlice<'a, T> {
    raw_data: &'a mut [MaybeUninit<T>],
    len: usize,
}

impl<'a, T> PartialInitSlice<'_, T> {
    fn create_by_split(
        buffer: &'a mut [MaybeUninit<u8>],
        cap: usize,
    ) -> (Self, &'a mut [MaybeUninit<u8>]) {
        const {
            assert!(align_of::<T>() <= std::mem::size_of::<T>());
        }
        let (this_buffer, buffer) = buffer.split_at_mut(cap * std::mem::size_of::<T>());
        (
            Self {
                // SAFETY:
                // we are casting `this_buffer` to `MaybeUninit<T>`
                // the size of `this_buffer` is `cap` times `size_of::<T>`
                // so the cast is safe
                raw_data: unsafe {
                    core::slice::from_raw_parts_mut(
                        this_buffer.as_mut_ptr() as *mut MaybeUninit<T>,
                        cap,
                    )
                },
                len: 0,
            },
            buffer,
        )
    }
    #[inline]
    fn push(&mut self, el: T) {
        self.raw_data[self.len].write(el);
        self.len += 1;
    }
    #[inline]
    fn get_slice(self) -> &'a [T] {
        assert!(self.len <= self.raw_data.len());
        // SAFETY: we only increment `self.len` after initializing the value at it's index
        // this function takes ownership of Self, so no one else can mutate `self.mem`
        unsafe { std::slice::from_raw_parts(self.raw_data.as_ptr() as *const T, self.len) }
    }
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
    #[inline]
    fn last(&self) -> Option<&T> {
        if self.len > 0 {
            // SAFETY: everything less then `self.len` is initialized
            Some(unsafe { self.raw_data[self.len - 1].assume_init_ref() })
        } else {
            None
        }
    }
    #[inline]
    fn copy_from_slice(&mut self, slice: &[T])
    where
        T: Copy,
    {
        // treat the slice as an uninit slice
        // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout
        let uninit_slice: &[MaybeUninit<T>] = unsafe { core::mem::transmute(slice) };

        self.raw_data[self.len..self.len + slice.len()].copy_from_slice(uninit_slice);

        self.len += slice.len();
    }
}

pub const fn split_1_2_alloc_size(input_len: usize, updates_len: usize) -> usize {
    let input_len = input_len * size_of::<usize>();
    let removes_len = updates_len * size_of::<usize>();
    let inserts_len = size_of::<(usize, u8)>();
    let inserted_len = input_len + inserts_len;
    removes_len + inserts_len + (inserted_len * 2) + input_len
}
pub fn update_split_new_types_1_2<'a, 'b>(
    input: &'a [u8],
    updates: &'a [Update],
    buffer: &'b mut [MaybeUninit<u8>],
) -> &'b [u8] {
    let (mut removes, buffer) = PartialInitSlice::<'b, _>::create_by_split(buffer, updates.len());
    let (mut inserts, buffer) = PartialInitSlice::<'b, _>::create_by_split(buffer, updates.len());
    for update in updates {
        match *update {
            Update::Remove(idx) => {
                let remove_idx = idx + inserts.len();
                if Some(&remove_idx) != removes.last() {
                    removes.push(idx + inserts.len())
                }
            }
            Update::Insert(idx, val) => inserts.push((idx, val)),
        }
    }

    let (mut inserted, buffer) =
        PartialInitSlice::<'b, _>::create_by_split(buffer, input.len() + inserts.len());
    let inserts = inserts.get_slice();
    let mut prev_idx = 0;
    for &(insert_idx, val) in inserts {
        inserted.copy_from_slice(&input[prev_idx..insert_idx]);
        inserted.push(val);
        prev_idx = insert_idx;
    }
    inserted.copy_from_slice(&input[prev_idx..]);
    let inserted = inserted.get_slice();

    let (mut output, _) =
        PartialInitSlice::<'b, _>::create_by_split(buffer, inserted.len() - removes.len());
    let mut prev_idx = 0;
    for &remove_idx in removes.get_slice() {
        output.copy_from_slice(&inserted[prev_idx..remove_idx]);
        prev_idx = remove_idx + 1;
    }
    output.copy_from_slice(&inserted[prev_idx..]);
    output.get_slice()
}

pub fn update_split_new_types_2(input: &[u8], updates: &[Update]) -> Vec<u8> {
    // indexes of removes
    let mut removes = Vec::<usize>::with_capacity(updates.len());
    // all of the data that will be inserted into
    let mut inserts = Vec::<u8>::with_capacity(updates.len());
    // list of the index to insert at and range to take from inserts
    let mut insert_idxs = Vec::<(usize, std::ops::Range<usize>)>::with_capacity(updates.len());

    for update in updates {
        match *update {
            Update::Remove(idx) => {
                let remove_idx = idx + inserts.len();
                if Some(&remove_idx) != removes.last() {
                    removes.push(idx + inserts.len())
                }
            }
            Update::Insert(idx, val) => {
                inserts.push(val);
                if let Some(last_idx) = insert_idxs.last_mut()
                    && last_idx.0 == idx
                {
                    last_idx.1.end += 1;
                } else {
                    insert_idxs.push((idx, (inserts.len() - 1)..inserts.len()));
                }
            }
        }
    }

    let mut inserted = Vec::with_capacity(input.len() + inserts.len());
    let mut prev_idx = 0;
    for (insert_idx, inserts_range) in insert_idxs {
        // copy over all the stuff that hasn't changed
        inserted.extend_from_slice(&input[prev_idx..insert_idx]);
        // copy over all of the inserts
        inserted.extend_from_slice(&inserts[inserts_range]);
        prev_idx = insert_idx;
    }
    inserted.extend_from_slice(&input[prev_idx..]);

    let mut output = Vec::with_capacity(inserted.len() - removes.len());
    let mut prev_idx = 0;
    for remove_idx in removes {
        output.extend_from_slice(&inserted[prev_idx..remove_idx]);
        prev_idx = remove_idx + 1;
    }
    output.extend_from_slice(&inserted[prev_idx..]);
    output
}

pub fn update_split_new_types_3(input: &[u8], updates: &[Update]) -> Vec<u8> {
    ContigUpdates::new(updates).updated_vec(input)
}

pub fn update_new_types_3_pre_compute(
    input: &[u8],
    contig_updates: &ContigUpdates,
    inserts_vec: &mut Vec<u8>,
    output: &mut Vec<u8>,
) {
    contig_updates.fill_inserts_vec(input, inserts_vec);
    contig_updates.fill_removes(inserts_vec, output);
}

#[inline]
pub fn update_in_chunks_gen<const CHUNK_SIZE: usize>(input: &[u8], updates: &[Update]) -> Vec<u8> {
    let chunks = input.chunks_exact(CHUNK_SIZE);
    let extra = chunks.remainder();

    let mut updates = updates.iter();
    let mut next_update = updates.next();

    let mut output = Vec::with_capacity(input.len() + updates.len());

    let mut offset: i64 = 0;
    let mut last_index = None;
    for (idx, chunk) in chunks.enumerate() {
        // start by copying the slice into the new array
        output.extend_from_slice(chunk);
        // then use the same approach from `simple_update` but it's cheaper since
        while let Some(update) = next_update {
            let index = update.offset_index(offset);
            if index > idx || index >= output.len() {
                break;
            }

            match *update {
                // idempotent removal
                Update::Remove(idx) if Some(idx) == last_index => (),
                Update::Remove(idx) => {
                    output.remove(index);
                    last_index = Some(idx);
                    offset -= 1;
                }
                Update::Insert(_, value) => {
                    output.insert(index, value);
                    offset += 1;
                }
            }
            next_update = updates.next();
        }
    }
    output.extend_from_slice(extra);
    // then use the same approach from `simple_update` but limited to this scope
    let mut last_index = None;
    while let Some(update) = next_update {
        let index = update.offset_index(offset);
        match *update {
            // idempotent removal
            Update::Remove(idx) if Some(idx) == last_index => (),
            Update::Remove(idx) => {
                output.remove(index);
                last_index = Some(idx);
                offset -= 1;
            }
            Update::Insert(_, value) => {
                output.insert(index, value);
                offset += 1;
            }
        }
        next_update = updates.next();
    }
    output
}

pub fn update_in_chunks(input: &[u8], updates: &[Update]) -> Vec<u8> {
    update_in_chunks_gen::<2>(input, updates)
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_update_simple() {
        let mut input = vec![1, 2, 3, 4, 5, 6, 7, 8];
        update_simple(
            &mut input,
            &[
                Update::Remove(0),
                Update::Insert(3, 5),
                Update::Insert(8, 3),
            ],
        );
        assert_eq!(input, vec![2, 3, 5, 4, 5, 6, 7, 8, 3]);

        let mut input = vec![1, 2];
        update_simple(&mut input, &[Update::Remove(0), Update::Remove(0)]);
        assert_eq!(input, vec![2]);

        let mut input = vec![150, 52, 115, 80, 31, 94, 151, 56, 164, 205];
        update_simple(
            &mut input,
            &[
                Update::Remove(4),
                Update::Insert(5, 98),
                Update::Remove(5),
                Update::Insert(8, 183),
                Update::Remove(8),
            ],
        );
        assert_eq!(input, vec![150, 52, 115, 80, 98, 151, 56, 183, 205]);

        let mut input = vec![187, 238, 254, 179, 152];
        update_simple(
            &mut input,
            &[
                Update::Insert(1, 253),
                Update::Remove(1),
                Update::Insert(2, 4),
            ],
        );
        assert_eq!(input, vec![187, 253, 4, 254, 179, 152]);

        let mut input = vec![38, 78, 5, 173, 9];
        update_simple(
            &mut input,
            &[Update::Insert(3, 70), Update::Remove(3), Update::Remove(4)],
        );
        assert_eq!(input, vec![38, 78, 5, 70]);
    }

    #[test]
    #[should_panic]
    fn test_update_simple_panics() {
        let mut input = vec![1, 2, 3, 4, 5, 6, 7, 8];
        update_simple(&mut input, &[Update::Insert(9, 3)]);
    }

    #[test]
    fn test_update_realloc() {
        let mut input = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let input1 = input.clone();
        let updates = &[
            Update::Remove(0),
            Update::Insert(3, 3),
            Update::Insert(8, 3),
        ];
        update_simple(&mut input, updates);
        assert_eq!(input, update_realloc(&input1, updates));
        for _ in 0..1_000 {
            let (mut input, updates) = build_both(5, 10);
            let input1 = input.clone();
            update_simple(&mut input, &updates);
            assert_eq!(input, update_realloc(&input1, &updates));
        }
    }

    #[test]
    fn test_update_iterator() {
        let updates = &[
            Update::Insert(0, 1),
            Update::Insert(0, 2),
            Update::Insert(0, 3),
            Update::Insert(0, 4),
        ];
        assert_eq!(vec![1, 2, 3, 4], update_collect_iter(&[], updates));
        for _ in 0..1_000 {
            let (mut input, updates) = build_both(5, 10);
            let input1 = input.clone();
            update_simple(&mut input, &updates);
            assert_eq!(input, update_collect_iter(&input1, &updates));
        }
    }

    #[test]
    fn test_inserts_update_split() {
        let updates = &[
            Update::Insert(0, 1),
            Update::Insert(0, 2),
            Update::Insert(0, 3),
            Update::Insert(0, 4),
        ];
        assert_eq!(vec![1, 2, 3, 4], update_split_alloc(&[], updates));
    }
    #[test]
    fn test_removes_update_split() {
        let updates = &[
            Update::Remove(0),
            Update::Remove(0),
            Update::Remove(2),
            Update::Remove(3),
        ];
        assert_eq!(vec![2], update_split_alloc(&[1, 2, 3, 4], updates));
    }
    #[test]
    fn test_update_split_alloc() {
        for _ in 0..1_000 {
            let (mut input, updates) = build_both(5, 10);
            println!("input: {input:?}\nupdates: {updates:?}");
            let input1 = input.clone();
            update_simple(&mut input, &updates);
            assert_eq!(input, update_split_alloc(&input1, &updates));
        }
    }
    #[test]
    fn test_update_split_new_type() {
        for _ in 0..1_000 {
            let (mut input, updates) = build_both(5, 10);
            println!("input: {input:?}\nupdates: {updates:?}");
            let input1 = input.clone();
            update_simple(&mut input, &updates);
            assert_eq!(input, update_split_new_types(&input1, &updates));
        }
    }
    #[test]
    fn test_update_split_new_type_1() {
        for _ in 0..1_000 {
            let (mut input, updates) = build_both(5, 10);
            println!("input: {input:?}\nupdates: {updates:?}");
            let input1 = input.clone();
            update_simple(&mut input, &updates);
            assert_eq!(input, update_split_new_types_1(&input1, &updates));
        }
    }
    #[test]
    fn test_update_split_new_type_1_1() {
        for _ in 0..1_000 {
            let (mut input, updates) = build_both(50, 100);
            println!("input: {input:?}\nupdates: {updates:?}");
            let input1 = input.clone();
            update_simple(&mut input, &updates);
            assert_eq!(input, update_split_new_types_1_1(&input1, &updates));
        }
    }
    #[test]
    fn test_update_split_new_type_1_2() {
        for _ in 0..1_000 {
            let (mut input, updates) = build_both(50, 100);
            println!("input: {input:?}\nupdates: {updates:?}");
            let input1 = input.clone();
            let mut buffer =
                Box::new_uninit_slice(split_1_2_alloc_size(input.len(), updates.len()));
            update_simple(&mut input, &updates);
            assert_eq!(
                input,
                update_split_new_types_1_2(&input1, &updates, &mut buffer)
            );
        }
    }
    #[test]
    fn test_update_split_new_type_2() {
        for _ in 0..1_000 {
            let (mut input, updates) = build_both(5, 10);
            println!("input: {input:?}\nupdates: {updates:?}");
            let input1 = input.clone();
            update_simple(&mut input, &updates);
            assert_eq!(input, update_split_new_types_2(&input1, &updates));
        }
    }
    #[test]
    fn test_update_split_new_type_3() {
        for _ in 0..1_000 {
            let (mut input, updates) = build_both(500, 1_000);
            println!("input: {input:?}\nupdates: {updates:?}");
            let input1 = input.clone();
            update_simple(&mut input, &updates);
            assert_eq!(input, update_split_new_types_3(&input1, &updates));
        }
    }

    #[test]
    fn test_update_new_types_1_pre_compute() {
        for _ in 0..1_000 {
            let (mut input, updates) = build_both(500, 1_000);
            let input1 = input.clone();
            println!("input: {input:?}\nupdates: {updates:?}");

            let (removes, inserts) = init_new_types_1(&updates);
            let mut inserts_vec = Vec::with_capacity(input1.len() + inserts.len());

            let mut output = Vec::with_capacity(input1.len() + inserts.len() - removes.len());

            update_new_types_1_pre_compute(
                &input1,
                &removes,
                &inserts,
                &mut inserts_vec,
                &mut output,
            );

            update_simple(&mut input, &updates);

            assert_eq!(input, output);
        }
    }

    #[test]
    fn test_update_new_types_3_pre_compute() {
        for _ in 0..1_000 {
            let (mut input, updates) = build_both(500, 1_000);
            let input1 = input.clone();
            println!("input: {input:?}\nupdates: {updates:?}");

            let contig_updates = ContigUpdates::new(&updates);

            let mut inserts_vec = contig_updates.alloc_inserts_vec(input1.len());
            let mut output = contig_updates.alloc_removes_vec(inserts_vec.capacity());

            update_new_types_3_pre_compute(&input1, &contig_updates, &mut inserts_vec, &mut output);

            update_simple(&mut input, &updates);

            assert_eq!(input, output);
        }
    }

    #[test]
    fn test_update_in_chunks() {
        for _ in 0..10_000 {
            let (mut input, updates) = build_both(50, 100);
            println!("input: {input:?}\nupdates: {updates:?}");
            let input1 = input.clone();
            update_simple(&mut input, &updates);
            assert_eq!(input, update_in_chunks_gen::<3>(&input1, &updates));
        }
    }
}
