//! all of the functions update the inputs based on the slice of updates

#[derive(Clone, Copy, PartialEq, Eq)]
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
        self.index().cmp(&other.index())
    }
}
impl PartialOrd for Update {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// basic for loop implementation: go through all updates and apply them
/// used as a baseline test for all other implementations
pub fn update_simple(input: &mut Vec<u8>, updates: &[Update]) {
    let mut offset: i64 = 0;
    for update in updates {
        let index = update.offset_index(offset);
        match *update {
            Update::Remove(_) => {
                input.remove(index);
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
pub fn update_realloc(input: &mut Vec<u8>, updates: &[Update]) {
    // calculate the change in the vector size
    let total_offset: i64 = updates
        .iter()
        .map(|update| match update {
            Update::Remove(_) => -1,
            Update::Insert(_, _) => 1,
        })
        .sum();
    let v = Vec::with_capacity(input.len() + total_offset as usize);
    let mut updates = updates.iter();
    let (update, v) =
        input
            .iter()
            .enumerate()
            .fold((updates.next(), v), |(mut update, mut v), (idx, &val)| {
                // loop until no updates apply to current idx
                loop {
                    match update {
                        Some(Update::Remove(index)) if *index == idx => {
                            update = updates.next();
                            break;
                        }
                        Some(Update::Insert(index, insert_val)) if *index == idx => {
                            v.push(*insert_val);
                            update = updates.next();
                            continue;
                        }
                        _ => {
                            v.push(val);
                            break;
                        }
                    }
                }
                (update, v)
            });
    *input = v;

    // if there are any updates left, apply them
    for update in update.into_iter().chain(updates) {
        match update {
            Update::Remove(_) => {}
            Update::Insert(_, val) => input.push(*val),
        }
    }
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
                Update::Insert(3, 3),
                Update::Insert(8, 3),
            ],
        );
        assert_eq!(input, vec![2, 3, 3, 4, 5, 6, 7, 8, 3]);

        let mut input = vec![1, 2];
        update_simple(&mut input, &[Update::Remove(0), Update::Remove(0)]);
        assert_eq!(input, vec![]);
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
        let mut input1 = input.clone();
        let updates = &[
            Update::Remove(0),
            Update::Insert(3, 3),
            Update::Insert(8, 3),
        ];
        update_simple(&mut input, updates);
        update_realloc(&mut input1, updates);
        assert_eq!(input, input1);
    }
}
