//! all of the functions update the inputs based on the slice of updates

#[derive(Clone, Copy)]
pub enum Update {
    Remove { index: usize },
    Insert { index: usize, value: u8 },
}

/// basic for loop implementation: go through all updates and apply them
/// used as a baseline test for all other implementations
pub fn update_simple(input: &mut Vec<u8>, updates: &[Update]) {
    for update in updates {
        match *update {
            Update::Remove { index } => {
                input.remove(index);
            }
            Update::Insert { index, value } => {
                input.insert(index, value);
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
                Update::Remove { index: 0 },
                Update::Insert { index: 3, value: 3 },
                Update::Insert { index: 8, value: 3 },
            ],
        );
        assert_eq!(input, vec![2, 3, 4, 3, 5, 6, 7, 8, 3])
    }

    #[test]
    #[should_panic]
    fn test_update_simple_panics() {
        let mut input = vec![1, 2, 3, 4, 5, 6, 7, 8];
        update_simple(&mut input, &[Update::Insert { index: 9, value: 3 }]);
    }
}
