use std::mem::MaybeUninit;

/// newtype wrapper that helps memory get layed out like:
///
/// [removes    ] [inserts        ] [insert_idxs] [insert_ranges  ]
/// [updates_len] [updates_len / 8] [updates_len] [updates_len + 1]
#[derive(Debug, Copy, Clone)]
struct UpdatesLen(usize);
impl UpdatesLen {
    #[inline]
    const fn removes_start(self) -> usize {
        0
    }
    #[inline]
    const fn removes_cap(self) -> usize {
        self.0
    }
    #[inline]
    const fn inserts_start(self) -> usize {
        self.removes_cap()
    }
    /// values take up less space then indexes
    /// specifically there are 8 values for each index
    #[inline]
    const fn inserts_cap(self) -> usize {
        self.0.div_ceil(8)
    }
    #[inline]
    const fn idxs_start(self) -> usize {
        self.removes_cap() + self.inserts_cap()
    }
    #[inline]
    const fn idxs_cap(self) -> usize {
        self.0
    }
    #[inline]
    const fn ranges_start(self) -> usize {
        self.removes_cap() + self.inserts_cap() + self.idxs_cap()
    }
    #[inline]
    const fn ranges_cap(self) -> usize {
        self.0 + 1
    }
}

/// contiguously allocated memory for updates
pub struct ContigUpdates {
    updates_len: UpdatesLen,
    /// actual memory
    /// setup like this:
    /// [removes    ] [inserts        ] [insert_idxs] [insert_ranges  ]
    /// [updates_len] [updates_len / 8] [updates_len] [updates_len + 1]
    memory: Box<[MaybeUninit<[u8; 8]>]>,
    /// len of removes list
    /// # Safety
    /// Only increment if the all values in `removes` are initialized
    /// must remain <= `removes_cap`
    removes_len: usize,
    /// len of inserts list
    /// # Safety
    /// Only increment if the all values in `inserts` are initialized
    /// This includes the entirity of the array indexed into
    /// so if `inserts_len == 3` then `memory[inserts_start] = [x, x, x, 0, 0, 0, 0, 0]`
    /// must remain <= `inserts_cap`
    inserts_len: usize,
    /// len of the index to insert
    /// # Safety
    /// Only increment if the all values in `idxs` are initialized
    /// must remain <= `idxs_cap`
    idxs_len: usize,
    /// list of offsets of where to get range
    /// # Safety
    /// Only increment if the all values in `ranges` are initialized
    /// must remain <= `ranges_cap`
    ranges_len: usize,
}

impl ContigUpdates {
    #[inline]
    fn new_uninit(updates_len: usize) -> Self {
        let updates_len = UpdatesLen(updates_len);
        let mut memory = Box::new_uninit_slice(
            updates_len.removes_cap()
                + updates_len.inserts_cap()
                + updates_len.idxs_cap()
                + updates_len.ranges_cap(),
        );

        // start by pushing 0 `ranges`
        memory[updates_len.ranges_start()].write(0_usize.to_ne_bytes());

        Self {
            updates_len,
            memory,
            removes_len: 0,
            inserts_len: 0,
            idxs_len: 0,
            ranges_len: 1,
        }
    }

    /// # Safety:
    /// it is the responsability of the caller that all bytes in the range
    /// are initialized, and that each array was initialized using `usize::to_ne_bytes`
    #[inline]
    unsafe fn assume_usize_slice(&self, start: usize, len: usize) -> &[usize] {
        let slice = &self.memory[start..start + len];
        // SAFETY:
        // the caller makes sure that each element in each array is initialized as usize
        let init_slice = unsafe { slice.assume_init_ref() };
        bytemuck::cast_slice(init_slice)
    }

    #[inline]
    fn inserts(&self) -> &[u8] {
        let inserts_start = self.updates_len.inserts_start();
        let inserts_arr_len = self.inserts_len.div_ceil(8);
        let slice = &self.memory[inserts_start..inserts_start + inserts_arr_len];
        // SAFETY:
        // `slice` is entirely initialized
        // - `inserts` start at removes_cap
        // - whenever we "push" to inserts we initialize all of the members of the array
        // - we only increment `self.inserts_len` after initializing the array it points into
        let init_slice = unsafe { slice.assume_init_ref() };
        // after casting to from [[u8]] to [u8], take only the bytes we are using
        &bytemuck::cast_slice(init_slice)[..self.inserts_len]
    }
    #[inline]
    fn push_remove(&mut self, idx: usize) {
        let remove_idx = (idx + self.inserts_len).to_ne_bytes();
        // SAFETY: accessing removes
        if self.removes_len == 0
            || unsafe {
                self.memory[self.updates_len.removes_start() + self.removes_len - 1].assume_init()
            } != remove_idx
        {
            assert!(
                self.removes_len < self.updates_len.removes_cap(),
                "trying `push_remove` past the end of `removes`"
            );
            self.memory[self.removes_len].write(remove_idx);
            self.removes_len += 1;
        }
    }
    /// only push `val` onto `inserts`
    #[inline]
    fn inserts_push(&mut self, val: u8) {
        assert!(self.inserts_len < self.updates_len.inserts_cap() * 8);

        let inserts_rem = self.inserts_len % 8;
        let inserts_idx = self.updates_len.inserts_start() + self.inserts_len / 8;
        let chunk_ref = &mut self.memory[inserts_idx];
        if inserts_rem == 0 {
            // initialize the entire chunk at once
            chunk_ref.write([val, 0, 0, 0, 0, 0, 0, 0]);
        } else {
            // SAFETY: we initialize the entire chunk at the same time
            let chunk = unsafe { chunk_ref.assume_init_mut() };
            chunk[inserts_rem] = val;
        }

        // SAFETY: either value was initialized in a previous call or just now
        #[allow(unused_unsafe)]
        unsafe {
            self.inserts_len += 1;
        }
    }

    /// push `val` onto `inserts` and update book keeping for `idx`
    #[inline]
    fn push_insert_index(&mut self, idx: usize, val: u8) {
        self.inserts_push(val);

        let idx = idx.to_ne_bytes();

        let push_idx = self.idxs_len == 0 || {
            // SAFETY: we are reading the last value of `insert_idxs`
            let last_insert_idx = unsafe {
                self.memory[self.updates_len.idxs_start() + self.idxs_len - 1].assume_init()
            };
            last_insert_idx != idx
        };

        if push_idx {
            assert!(self.idxs_len < self.updates_len.idxs_cap());
            self.memory[self.updates_len.idxs_start() + self.idxs_len].write(idx);
            // SAFETY: we initialized `idxs[idxs_len]` just now
            #[allow(unused_unsafe)]
            unsafe {
                self.idxs_len += 1;
            }

            assert!(self.ranges_len < self.updates_len.ranges_cap());
            self.memory[self.updates_len.ranges_start() + self.ranges_len]
                .write(self.inserts_len.to_ne_bytes());

            // SAFETY: we just initialized `ranges[ranges_len]` just now
            #[allow(unused_unsafe)]
            unsafe {
                self.ranges_len += 1;
            }
        } else {
            self.memory[self.updates_len.ranges_start() + self.ranges_len - 1]
                .write((self.inserts_len).to_ne_bytes());
        }
    }

    /// creates the contigously allocated form of the updates
    #[inline]
    pub fn new(updates: &[super::Update]) -> Self {
        let mut this = Self::new_uninit(updates.len());
        for update in updates {
            match *update {
                crate::Update::Remove(remove_idx) => this.push_remove(remove_idx),
                crate::Update::Insert(insert_idx, val) => this.push_insert_index(insert_idx, val),
            }
        }
        this
    }
    pub fn alloc_inserts_vec(&self, input_len: usize) -> Vec<u8> {
        Vec::with_capacity(input_len + self.inserts_len)
    }
    pub fn alloc_removes_vec(&self, input_len: usize) -> Vec<u8> {
        if self.removes_len < input_len {
            Vec::with_capacity(input_len - self.removes_len)
        } else {
            Vec::new()
        }
    }
    #[inline]
    pub(super) fn fill_inserts_vec(&self, input: &[u8], output: &mut Vec<u8>) {
        assert!(output.capacity() >= input.len() + self.inserts_len);

        output.clear();
        let inserts = self.inserts();

        let insert_idxs =
            unsafe { self.assume_usize_slice(self.updates_len.idxs_start(), self.idxs_len) };

        let insert_ranges =
            unsafe { self.assume_usize_slice(self.updates_len.ranges_start(), self.ranges_len) };
        let mut curr_idx = 0;
        for (insert_range, &insert_idx) in insert_ranges.windows(2).zip(insert_idxs) {
            let prev_idx = curr_idx;
            curr_idx = insert_idx;
            output.extend_from_slice(&input[prev_idx..insert_idx]);
            output.extend_from_slice(&inserts[insert_range[0]..insert_range[1]]);
        }
        output.extend_from_slice(&input[curr_idx..]);
    }
    #[inline]
    pub(super) fn inserts_vec(&self, input: &[u8]) -> Vec<u8> {
        let mut output = self.alloc_inserts_vec(input.len());
        self.fill_inserts_vec(input, &mut output);
        output
    }

    #[inline]
    pub(super) fn fill_removes(&self, input: &[u8], output: &mut Vec<u8>) {
        assert!(output.capacity() >= input.len() - self.removes_len);

        // SAFETY: accessing removes
        let removes =
            unsafe { self.assume_usize_slice(self.updates_len.removes_start(), self.removes_len) };

        let mut prev_idx = 0;
        for &remove_idx in removes {
            output.extend_from_slice(&input[prev_idx..remove_idx]);
            prev_idx = remove_idx + 1;
        }
        output.extend_from_slice(&input[prev_idx..]);
    }

    #[inline]
    pub(super) fn removed_vec(&self, input: &[u8]) -> Vec<u8> {
        let mut output = self.alloc_removes_vec(input.len());
        self.fill_removes(input, &mut output);
        output
    }

    #[inline]
    pub(super) fn updated_vec(&self, input: &[u8]) -> Vec<u8> {
        self.removed_vec(&self.inserts_vec(input))
    }
}
