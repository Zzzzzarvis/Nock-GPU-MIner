use nockvm::interpreter::Context;
use nockvm::jets::util::slot;
use nockvm::jets::JetErr;
use nockvm::noun::{Atom, Noun, D, T};

use crate::form::math::tip5::*;
use crate::jets::utils::jet_err;

pub fn hoon_list_to_sponge(list: Noun) -> Result<[u64; STATE_SIZE], JetErr> {
    if list.is_atom() {
        return jet_err();
    }

    let mut sponge = [0; STATE_SIZE];
    let mut current = list;
    let mut i = 0;

    while current.is_cell() {
        let cell = current.as_cell()?;
        sponge[i] = cell.head().as_atom()?.as_u64()?;
        current = cell.tail();
        i = i + 1;
    }

    if i != STATE_SIZE {
        return jet_err();
    }

    Ok(sponge)
}

pub fn vec_to_hoon_list(context: &mut Context, vec: &[u64]) -> Noun {
    let mut list = D(0);
    for e in vec.iter().rev() {
        let n = Atom::new(&mut context.stack, *e).as_noun();
        list = T(&mut context.stack, &[n, list]);
    }
    list
}

pub fn permutation_jet(context: &mut Context, subject: Noun) -> Result<Noun, JetErr> {
    let sample = slot(subject, 6)?;
    let mut sponge = hoon_list_to_sponge(sample)?;
    permute(&mut sponge);

    let new_sponge = vec_to_hoon_list(context, &sponge);

    Ok(new_sponge)
}
