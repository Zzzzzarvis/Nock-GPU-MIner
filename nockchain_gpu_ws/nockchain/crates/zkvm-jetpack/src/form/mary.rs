use super::Felt;

#[derive(Clone, PartialEq)]
pub struct Mary {
    pub step: u32,
    pub len: u32,
    pub dat: Vec<u64>,
}

#[derive(Clone, Copy)]
pub struct MarySlice<'a> {
    pub step: u32,
    pub len: u32,
    pub dat: &'a [u64],
}

pub struct MarySliceMut<'a> {
    pub step: u32,
    pub len: u32,
    pub dat: &'a mut [u64],
}

pub struct Table<'a> {
    pub num_cols: u32,
    pub mary: MarySlice<'a>,
}

impl Mary {
    pub fn as_slice(&self) -> MarySlice {
        MarySlice {
            step: self.step,
            len: self.len,
            dat: self.dat.as_slice(),
        }
    }
    pub fn as_mut_slice(&mut self) -> MarySliceMut {
        MarySliceMut {
            step: self.step,
            len: self.len,
            dat: self.dat.as_mut_slice(),
        }
    }
}

impl TryFrom<MarySlice<'_>> for &[Felt] {
    type Error = ();

    #[inline(always)]
    fn try_from(m: MarySlice) -> std::result::Result<Self, Self::Error> {
        assert_eq!(m.step, 3);

        let dat_slice: &[Felt] =
            unsafe { std::slice::from_raw_parts(m.dat.as_ptr() as *const Felt, m.len as usize) };
        Ok(dat_slice)
    }
}

impl TryFrom<MarySliceMut<'_>> for &[Felt] {
    type Error = ();

    #[inline(always)]
    fn try_from(m: MarySliceMut) -> std::result::Result<Self, Self::Error> {
        assert_eq!(m.step, 3);

        let dat_slice: &[Felt] = unsafe {
            std::slice::from_raw_parts(<[u64]>::as_ptr(&m.dat[0..3]) as *const Felt, m.len as usize)
        };
        Ok(dat_slice)
    }
}

impl TryFrom<MarySliceMut<'_>> for &mut [Felt] {
    type Error = ();

    #[inline(always)]
    fn try_from(m: MarySliceMut) -> std::result::Result<Self, Self::Error> {
        assert_eq!(m.step, 3);

        let dat_slice: &mut [Felt] = unsafe {
            std::slice::from_raw_parts_mut(
                <[u64]>::as_mut_ptr(&mut m.dat[0..3]) as *mut Felt,
                m.len as usize,
            )
        };
        Ok(dat_slice)
    }
}

impl std::fmt::Debug for Mary {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Mary: (step={}, len={}, dat={:?})\r",
            self.step, self.len, self.dat
        )
    }
}

impl std::fmt::Debug for MarySlice<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "MarySlice: (step={}, len={}, dat={:?})\r",
            self.step, self.len, self.dat
        )
    }
}

impl std::fmt::Debug for MarySliceMut<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "MarySliceMut: (step={}, len={}, dat={:?})\r",
            self.step, self.len, self.dat
        )
    }
}
