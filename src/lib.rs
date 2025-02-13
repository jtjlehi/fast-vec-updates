//! all of the functions update the inputs based on the slice of updates

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Update {
    Remove(i64),
    Insert(i64, u8),
}

impl Update {
    #[inline]
    fn index(self, offset: i64) -> usize {
        let idx = match self {
            Update::Remove(index) => index + offset,
            Update::Insert(index, _) => index + offset,
        };
        if idx < 0 { 0 } else { idx as usize }
    }
}

impl Ord for Update {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.index(0).cmp(&other.index(0))
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
        let index = update.index(offset);
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
}
