use std::cmp;
use usiagent::rule::{LegalMove, Rule, Square, State};
use usiagent::shogi::KomaKind::{SFu, SKaku, SHisha, GFu, GKaku, GHisha};
use usiagent::shogi::{KomaKind, ObtainKind, Teban};

#[inline]
pub fn attack_priority(teban:Teban,state:&State,m:LegalMove) -> i32 {
    const KPT_VALUES:[i32;29] = [
        1, 2, 2, 3, 5, 5, 5, 8, 5, 5, 5, 5, 8, 8,
        1, 2, 2, 3, 5, 5, 5, 8, 5, 5, 5, 5, 8, 8,
        0
    ];

    let mut value = 0;

    match m {
        LegalMove::To(m) if !m.is_nari() => {
            let x = m.src() / 9;
            let y = m.src() - x * 9;

            let kind = state.get_banmen().0[y as usize][x as usize];

            match kind {
                SFu | SKaku | SHisha |
                GFu | GKaku | GHisha if Rule::is_possible_nari(kind,m.src() as Square,m.dst() as Square) => {
                    value += 100;
                },
                _ => ()
            }
        },
        _ => ()
    }

    let kind = match m {
        LegalMove::To(m) if m.is_nari() => {
            let x = m.src() / 9;
            let y = m.src() - x * 9;

            state.get_banmen().0[y as usize][x as usize].to_nari() as usize
        },
        LegalMove::To(m) => {
            let x = m.src() / 9;
            let y = m.src() - x * 9;

            state.get_banmen().0[y as usize][x as usize] as usize
        },
        LegalMove::Put(m) => {
            m.kind() as usize
        }
    };

    value -= KPT_VALUES[kind];

    match m {
        LegalMove::To(m) => {
            value += dist(Rule::ou_square(teban,state),m.dst() as Square);
        },
        LegalMove::Put(m) => {
            value += dist(Rule::ou_square(teban,state),m.dst() as Square);
        }
    }

    value
}
#[inline]
pub fn defense_priority(_:Teban,state:&State,m:LegalMove) -> i32 {
    match m {
        LegalMove::Put(m) => {
            m.kind() as i32 + KomaKind::SHishaN as i32 + ObtainKind::HishaN as i32 + 4
        },
        LegalMove::To(m) if m.obtained() == Some(ObtainKind::Ou) => {
            0
        },
        LegalMove::To(m)=> {
            let src = m.src();
            let x = src / 9;
            let y = src - x * 9;
            let kind = state.get_banmen().0[y as usize][x as usize];

            match m.obtained() {
                Some(o) => {
                    o as i32 - ObtainKind::HishaN as i32 + 1
                },
                None if kind == KomaKind::SOu || kind == KomaKind::GOu => {
                    ObtainKind::HishaN as i32 + 2
                },
                None => {
                    ObtainKind::HishaN as i32 + 3
                }
            }
        }
    }
}
#[inline]
fn dist(o:Square,to:Square) -> i32 {
    let ox = o / 9;
    let oy = o - ox * 9;
    let tx = to / 9;
    let ty = to - tx * 9;

    cmp::max((ox - tx).abs(),(oy - ty).abs())
}