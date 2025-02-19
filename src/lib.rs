//! all of the functions update the inputs based on the slice of updates

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

    // if there are any updates left, apply them
    // for update in update.into_iter().chain(updates) {
    //     match update {
    //         Update::Remove(_) => {}
    //         Update::Insert(_, val) => v.push(*val),
    //     }
    // }
    v
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
}
