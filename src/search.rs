use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::marker::PhantomData;
use std::ops::{Add, Neg, Sub};
use std::sync::{Arc, atomic, mpsc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};
use concurrent_fixed_hashmap::ConcurrentFixedHashMap;
use usiagent::command::{UsiInfoSubCommand, UsiScore, UsiScoreMate};
use usiagent::error::EventHandlerError;
use usiagent::event::{EventDispatcher, MapEventKind, UserEvent, UserEventDispatcher, UserEventKind, UserEventQueue, USIEventDispatcher};
use usiagent::hash::{KyokumenHash, KyokumenMap};
use usiagent::logger::Logger;
use usiagent::OnErrorHandler;
use usiagent::player::InfoSender;
use usiagent::rule::{LegalMove, Rule, State};
use usiagent::shogi::{MochigomaCollections, MochigomaKind, ObtainKind, Teban};
use crate::error::{ApplicationError};
use crate::nn::Evalutor;
use crate::search::Score::{INFINITE, NEGINFINITE};
use crate::solver::{GameStateForMate, MaybeMate, Solver};

pub const BASE_DEPTH:u32 = 2;
pub const MAX_DEPTH:u32 = 6;
pub const TIMELIMIT_MARGIN:u64 = 50;
pub const NETWORK_DELAY:u32 = 1100;
pub const DEFALUT_DISPLAY_EVALUTE_SCORE:bool = false;
pub const MAX_THREADS:u32 = 1;
pub const MAX_PLY:u32 = 200;
pub const MAX_PLY_TIMELIMIT:u64 = 0;
pub const TURN_COUNT:u32 = 50;
pub const MIN_TURN_COUNT:u32 = 5;
pub const DEFAULT_STRICT_MATE:bool = true;
pub const DEFAULT_MATE_HASH:usize = 8;

pub trait Search<L,S>: Sized where L: Logger + Send + 'static, S: InfoSender {
    fn search<'a,'b>(&self,env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                     node: &mut GameNode,
                     event_dispatcher:&mut UserEventDispatcher<'b,Self,ApplicationError,L>,
                     evalutor: &Evalutor) -> Result<EvaluationResult,ApplicationError>;

    fn timelimit_reached(&self,env:&mut Environment<L,S>) -> bool {
        let network_delay = env.network_delay;
        env.limit.map_or(false,|l| {
            l < Instant::now() || l - Instant::now() <= Duration::from_millis(network_delay as u64 + TIMELIMIT_MARGIN)
        })
    }

    fn timeout_expected(&self, env:&mut Environment<L,S>) -> bool {
        env.current_limit.map(|l| Instant::now() >= l).unwrap_or(false)
    }

    fn end_of_search(&self, env:&mut Environment<L,S>) -> bool {
        env.turn_limit.map(|l| Instant::now() >= l).unwrap_or(false)
    }

    fn send_message(&self, env:&mut Environment<L,S>, message:&str) -> Result<(),ApplicationError>
        where Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
        let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
        commands.push(UsiInfoSubCommand::Str(String::from(message)));

        Ok(env.info_sender.send(commands)?)
    }

    fn send_message_immediate(&self, env:&mut Environment<L,S>, message:&str) -> Result<(),ApplicationError>
        where Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
        let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
        commands.push(UsiInfoSubCommand::Str(String::from(message)));

        Ok(env.info_sender.send_immediate(commands)?)
    }

    fn send_depth(&self, env:&mut Environment<L,S>,
                            depth:u32, seldepth:u32) -> Result<(),ApplicationError> {

        let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
        commands.push(UsiInfoSubCommand::Depth(depth));

        Ok(env.info_sender.send(commands)?)
    }

    fn send_info(&self, env:&mut Environment<L,S>,
                      depth:u32, seldepth:u32, pv:&VecDeque<LegalMove>, score:&Score) -> Result<(),ApplicationError>
        where Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

        let mut commands: Vec<UsiInfoSubCommand> = Vec::new();

        match score {
            Score::Value(1.) => {
                commands.push(UsiInfoSubCommand::Score(UsiScore::Mate(UsiScoreMate::Plus)))
            },
            Score::Value(0.) => {
                commands.push(UsiInfoSubCommand::Score(UsiScore::Mate(UsiScoreMate::Minus)))
            },
            Score::Value(s) => {
                commands.push(UsiInfoSubCommand::Score(UsiScore::Cp(s.into())))
            }
        }
        if depth < seldepth {
            commands.push(UsiInfoSubCommand::Depth(depth));
            commands.push(UsiInfoSubCommand::SelDepth(seldepth));
        }

        if pv.len() > 0 {
            commands.push(UsiInfoSubCommand::CurrMove(pv[0].to_move()));
            commands.push(UsiInfoSubCommand::Pv(pv.clone().into_iter().map(|m| m.to_move()).collect()));
        }
        commands.push(UsiInfoSubCommand::Time((Instant::now() - env.think_start_time).as_millis() as u64));

        Ok(env.info_sender.send_immediate(commands)?)
    }

    fn send_score(&self,env:&mut Environment<L,S>,
                        teban:Teban,
                        s:Score) -> Result<(),ApplicationError>
        where Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

        if env.display_evalute_score {
            let teban_str = match teban {
                Teban::Sente => "sente",
                Teban::Gote =>  "gote"
            };
            match &s {
                Score::INFINITE => {
                    self.send_message(env,&format!("evalute score = inifinite. ({0})",teban_str))
                },
                Score::NEGINFINITE => {
                    self.send_message(env,&format!("evalute score = neginifinite. ({0})",teban_str))
                },
                Score::Value(s) => {
                    self.send_message(env,&format!("evalute score =  {0: >17} ({1})",s,teban_str))
                }
            }
        } else {
            Ok(())
        }
    }

    fn startup_strategy<'a>(&self,env:&mut Environment<L,S>,
                            gs: &mut GameState<'a>,
                            mhash:u64,
                            shash:u64,
                            m:LegalMove,
                            is_oute:bool)
                            -> Option<(Option<ObtainKind>,KyokumenMap<u64,()>,KyokumenMap<u64,u32>)> {
        let mut oute_kyokumen_map = gs.oute_kyokumen_map.clone();
        let mut current_kyokumen_map = gs.current_kyokumen_map.clone();

        if is_oute {
            match oute_kyokumen_map.get(gs.teban,&mhash,&shash) {
                Some(_) => {
                    return None;
                },
                None => {
                    oute_kyokumen_map.insert(gs.teban,mhash,shash,());
                },
            }
        }

        if !is_oute {
            oute_kyokumen_map.clear(gs.teban);
        }

        let is_sennichite = match current_kyokumen_map.get(gs.teban,&mhash,&shash).unwrap_or(&0) {
            &c if c >= 3 => {
                return None;
            },
            &c if c > 0 => {
                current_kyokumen_map.insert(gs.teban,mhash,shash,c+1);

                true
            },
            _ => false,
        };

        Some((obtained,oute_kyokumen_map,current_kyokumen_map))
    }

    fn before_search<'a,'b>(&self,
                         env: &mut Environment<L, S>,
                         gs: &mut GameState<'a>,
                         node:&mut GameNode,
                         event_dispatcher:&mut UserEventDispatcher<'b,Self,ApplicationError,L>,
                         evalutor: &Evalutor)
        -> Result<BeforeSearchResult, ApplicationError> {

        self.send_depth(env,gs.current_depth)?;

        if self.end_of_search(env) {
            return Ok(BeforeSearchResult::Terminate(None));
        } else if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
            return Ok(BeforeSearchResult::Complete(EvaluationResult::Timeout));
        }

        if let Some(ObtainKind::Ou) = gs.obtained {
            return Ok(BeforeSearchResult::Terminate(Some(Score::Value(0.))));
        }

        if let Some(m) = gs.m {
            if Rule::is_mate(gs.teban, &*gs.state) {
                let mut mvs = VecDeque::new();
                mvs.push_front(m);
                return Ok(BeforeSearchResult::Terminate(Some(Score::Value(1.))));
            }
        }

        let mvs = if Rule::is_mate(gs.teban.opposite(),&*gs.state) {
            let mvs = Rule::respond_oute_only_moves_all(gs.teban, &*gs.state, &*gs.mc);

            if mvs.len() == 0 {
                let mut mvs = VecDeque::new();
                gs.m.map(|m| mvs.push_front(m));

                return Ok(BeforeSearchResult::Complete(
                    return Ok(BeforeSearchResult::Complete(EvaluationResult::Terminate(Some(Score::Value(0.)))))
                ));
            } else {
                mvs
            }
        } else {
            let mvs:Vec<LegalMove> = Rule::legal_moves_all(gs.teban, &*gs.state, &*gs.mc);

            mvs
        };

        if self.end_of_search(env) {
            return Ok(BeforeSearchResult::Complete(EvaluationResult::Terminate(None)));
        } else if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
            return Ok(BeforeSearchResult::Complete(EvaluationResult::Timeout));
        }

        let mut await_mvs = Vec::with_capacity(mvs.len());

        for m in mvs {
            let o = match m {
                LegalMove::To(ref m) => {
                    m.obtained().and_then(|o| MochigomaKind::try_from(o).ok())
                },
                _ => None
            };

            let mhash = env.hasher.calc_main_hash(gs.mhash,gs.teban,gs.state.get_banmen(),gs.mc,m.to_applied_move(),&o);
            let shash = env.hasher.calc_sub_hash(gs.shash,gs.teban,gs.state.get_banmen(),gs.mc,m.to_applied_move(),&o);

            if !env.kyokumen_map.contains_key((gs.teban,mhash,shash)) {
                let (s,r) = mpsc::channel();
                let (state,mc,_) = Rule::apply_move_none_check(&gs.state, gs.teban, gs.mc, m.to_applied_move());

                evalutor.submit(gs.teban.opposite(),state.get_banmen(),&mc,m,s)?;

                env.kyokumen_map.insert_new((gs.teban,mhash,shash),(Score::Value(0.),state,mc));

                await_mvs.push((r,m));
            }
        }
        Ok(BeforeSearchResult::AsyncMvs(await_mvs))
    }
}
#[derive(Debug)]
pub enum EvaluationResult {
    Value(Score,u64,VecDeque<LegalMove>),
    Timeout,
}
#[derive(Debug)]
pub enum RootEvaluationResult {
    Value(GameNode,Score,u64,VecDeque<LegalMove>),
    Timeout(GameNode),
}
#[derive(Debug)]
pub enum BeforeSearchResult {
    Complete(EvaluationResult),
    Terminate(Option<Score>),
    AsyncMvs(Vec<(Receiver<(u64,u64,i32)>,LegalMove)>),
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Score {
    Value(f32)
}
impl Neg for Score {
    type Output = Score;

    fn neg(self) -> Score {
        match self {
            Score::Value(v) => Score::Value(1.-v),
        }
    }
}
impl Add for Score {
    type Output = Self;

    fn add(self, other:Score) -> Self::Output {
        match (self,other) {
            (Score::Value(l),Score::Value(r)) => Score::Value(l + r),
        }
    }
}
impl From<Score> for i64 {
    fn from(s: Score) -> Self {
        match s {
            Score::Value(s) => ((s - 0.5) * (1i32 << 23) as f32) as i64
        }
    }
}
impl Add<Score> for f32 {
    type Output = Self;

    fn add(self, rhs: Score) -> Self::Output {
        match rhs {
            Score::Value(s) => {
                self + s
            }
        }
    }
}
pub struct Environment<L,S> where L: Logger, S: InfoSender {
    pub event_queue:Arc<Mutex<UserEventQueue>>,
    pub info_sender:S,
    pub on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
    pub hasher:Arc<KyokumenHash<u64>>,
    pub limit:Option<Instant>,
    pub current_limit:Option<Instant>,
    pub turn_limit:Option<Instant>,
    pub turn_count:u32,
    pub min_turn_count:u32,
    pub base_depth:u32,
    pub max_depth:u32,
    pub max_nodes:Option<i64>,
    pub max_ply:Option<u32>,
    pub max_ply_mate:Option<u32>,
    pub max_ply_timelimit:Option<Duration>,
    pub network_delay:u32,
    pub display_evalute_score:bool,
    pub max_threads:u32,
    pub mate_hash:usize,
    pub stop:Arc<AtomicBool>,
    pub quited:Arc<AtomicBool>,
    pub kyokumen_map:Arc<ConcurrentFixedHashMap<(Teban,u64,u64),(Score,Arc<State>,Arc<MochigomaCollections>)>>,
    pub nodes:Arc<AtomicU64>,
    pub think_start_time:Instant
}
impl<L,S> Clone for Environment<L,S> where L: Logger, S: InfoSender {
    fn clone(&self) -> Self {
        Environment {
            event_queue:Arc::clone(&self.event_queue),
            info_sender:self.info_sender.clone(),
            on_error_handler:Arc::clone(&self.on_error_handler),
            hasher:Arc::clone(&self.hasher),
            limit:self.limit.clone(),
            current_limit:self.current_limit.clone(),
            turn_limit:self.turn_limit.clone(),
            turn_count:self.turn_count,
            min_turn_count:self.min_turn_count,
            base_depth:self.base_depth,
            max_depth:self.max_depth,
            max_nodes:self.max_nodes.clone(),
            max_ply:self.max_ply.clone(),
            max_ply_mate:self.max_ply_mate.clone(),
            max_ply_timelimit:self.max_ply_timelimit.clone(),
            network_delay:self.network_delay,
            display_evalute_score:self.display_evalute_score,
            max_threads:self.max_threads,
            mate_hash:self.mate_hash,
            stop:Arc::clone(&self.stop),
            quited:Arc::clone(&self.quited),
            kyokumen_map:self.kyokumen_map.clone(),
            nodes:Arc::clone(&self.nodes),
            think_start_time:self.think_start_time.clone()
        }
    }
}
impl<L,S> Environment<L,S> where L: Logger, S: InfoSender {
    pub fn new(event_queue:Arc<Mutex<UserEventQueue>>,
               info_sender:S,
               on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
               hasher:Arc<KyokumenHash<u64>>,
               think_start_time:Instant,
               limit:Option<Instant>,
               current_limit:Option<Instant>,
               turn_limit:Option<Instant>,
               turn_count:u32,
               min_turn_count:u32,
               base_depth:u32,
               max_depth:u32,
               max_nodes:Option<i64>,
               max_ply:Option<u32>,
               max_ply_mate:Option<u32>,
               max_ply_timelimit:Option<Duration>,
               network_delay:u32,
               display_evalute_score:bool,
               max_threads:u32,
               mate_hash:usize
    ) -> Environment<L,S> {
        let stop = Arc::new(AtomicBool::new(false));
        let quited = Arc::new(AtomicBool::new(false));

        Environment {
            event_queue:event_queue,
            info_sender:info_sender,
            on_error_handler:on_error_handler,
            hasher:hasher,
            think_start_time:think_start_time,
            limit:limit,
            current_limit:current_limit,
            turn_limit:turn_limit,
            turn_count:turn_count,
            min_turn_count:min_turn_count,
            base_depth:base_depth,
            max_depth:max_depth,
            max_nodes:max_nodes,
            max_ply:max_ply,
            max_ply_mate:max_ply_mate,
            max_ply_timelimit:max_ply_timelimit,
            network_delay:network_delay,
            display_evalute_score:display_evalute_score,
            max_threads:max_threads,
            mate_hash:mate_hash,
            stop:stop,
            quited:quited,
            kyokumen_map:Arc::new(ConcurrentFixedHashMap::with_size(1 << 22)),
            nodes:Arc::new(AtomicU64::new(0))
        }
    }
}
pub struct GameState<'a> {
    pub teban:Teban,
    pub state:&'a Arc<State>,
    pub m:Option<LegalMove>,
    pub mc:&'a Arc<MochigomaCollections>,
    pub obtained:Option<ObtainKind>,
    pub current_kyokumen_map:&'a KyokumenMap<u64,u32>,
    pub oute_kyokumen_map:&'a KyokumenMap<u64,()>,
    pub mhash:u64,
    pub shash:u64,
    pub depth:u32,
    pub current_depth:u32
}
pub struct GameNode {
    win:f32,
    nodes:u64,
    m:LegalMove,
    mhash:u64,
    shash:u64,
    childlren:BinaryHeap<GameNode>
}
impl GameNode {
    pub fn new(m:LegalMove,mhash:u64,shash:u64) -> GameNode {
        GameNode {
            win:0.,
            nodes:0,
            m:m,
            mhash:mhash,
            shash:shash,
            childlren:BinaryHeap::new()
        }
    }

    pub fn computed_score(&self) -> f32 {
        if self.nodes == 0 {
            1.
        } else {
            self.win / self.nodes as f32
        }
    }
}
impl Ord for GameNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl PartialOrd for GameNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.computed_score().partial_cmp(&other.computed_score()).map(|r| {
            r.then((self as *const GameNode).cmp(&(other as *const GameNode)))
        })
    }
}
impl Eq for GameNode {}
impl PartialEq for GameNode {
    fn eq(&self, other: &Self) -> bool {
        self as *const GameNode == other as *const GameNode
    }
}
pub struct Root<L,S> where L: Logger + Send + 'static, S: InfoSender {
    l:PhantomData<L>,
    s:PhantomData<S>,
    receiver:Receiver<Result<RootEvaluationResult, ApplicationError>>,
    sender:Sender<Result<RootEvaluationResult, ApplicationError>>
}
impl<L,S> Root<L,S> where L: Logger + Send + 'static, S: InfoSender {
    pub fn new() -> Root<L,S> {
        let(s,r) = mpsc::channel();

        Root {
            l:PhantomData::<L>,
            s:PhantomData::<S>,
            receiver:r,
            sender:s
        }
    }

    pub fn create_event_dispatcher<'a,T>(on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,stop:&Arc<AtomicBool>,quited:&Arc<AtomicBool>)
                                       -> UserEventDispatcher<'a,T,ApplicationError,L> {

        let mut event_dispatcher = USIEventDispatcher::new(&on_error_handler);

        {
            let stop = stop.clone();

            event_dispatcher.add_handler(UserEventKind::Stop, move |_,e| {
                match e {
                    &UserEvent::Stop => {
                        stop.store(true,atomic::Ordering::Release);
                        Ok(())
                    },
                    e => Err(EventHandlerError::InvalidState(e.event_kind())),
                }
            });
        }

        {
            let stop = stop.clone();
            let quited = quited.clone();

            event_dispatcher.add_handler(UserEventKind::Quit, move |_,e| {
                match e {
                    &UserEvent::Quit => {
                        quited.store(true,atomic::Ordering::Release);
                        stop.store(true,atomic::Ordering::Release);
                        Ok(())
                    },
                    e => Err(EventHandlerError::InvalidState(e.event_kind())),
                }
            });
        }

        event_dispatcher
    }

    pub fn termination<'a,'b>(&self,
                       env:&mut Environment<L,S>,
                       gs: &mut GameState<'a>,
                       node:&mut GameNode,
                       evalutor: &Evalutor,
                       score:Score) -> Result<(),ApplicationError> {
        env.stop.store(true,atomic::Ordering::Release);

        while evalutor.active_threads() > 0 {
            match self.receiver.recv().map_err(|e| ApplicationError::from(e)).and_then(|r| r)? {
                RootEvaluationResult::Value(mut n,s, nodes, mvs) => {
                    n.nodes += nodes;
                    n.win += s;

                    node.childlren.push(n);
                },
                _ => ()
            }
            evalutor.on_end_thread().map_err(|e| ApplicationError::from(e))?;
        }

        Ok(())
    }

    fn parallelized<'a,'b>(&self,env:&mut Environment<L,S>,
                           gs:&mut GameState<'a>,
                           node:&mut GameNode,
                           event_dispatcher:&mut UserEventDispatcher<'b,Root<L,S>,ApplicationError,L>,
                           evalutor: &Evalutor) -> Result<EvaluationResult,ApplicationError>  {
        let mut gs = gs;

        let await_mvs = match self.before_search(env,&mut gs,event_dispatcher,node,evalutor)? {
            BeforeSearchResult::Complete(r) => {
                return Ok(r);
            },
            BeforeSearchResult::Terminate(None) => {
                return Ok(EvaluationResult::Value(Score::Value(0.),0,mvs));
            },
            BeforeSearchResult::Terminate(Some(s)) => {
                return Ok(EvaluationResult::Value(s,1,mvs));
            },
            BeforeSearchResult::AsyncMvs(mvs) => {
                mvs
            }
        };

        if await_mvs.len() > 0 {
            evalutor.begin_transaction()?;
        }

        let mut mvs = Vec::with_capacity(await_mvs.len());

        for r in await_mvs {
            let (mhash,shash,s) = r.0.recv()?;
            env.nodes.fetch_add(1,atomic::Ordering::Release);

            if let Some(mut g) = env.kyokumen_map.get_mut((gs.teban,mhash,shash)) {
                let (ref mut score,_,_) = *g;

                *score = s;
            }
        }

        let mvs_count = mvs.len() as u64;

        let mut threads = env.max_threads.min(mvs_count as u32);

        let sender = self.sender.clone();

        let mut is_timeout = false;

        loop {
            if threads == 0 {
                let r = self.receiver.recv();

                evalutor.on_end_thread()?;

                let r = r?.map_err(|e| ApplicationError::from(e))?;

                threads += 1;

                match r {
                    RootEvaluationResult::Value(mut n, win, nodes,  mvs) => {
                        n.win += win;
                        n.nodes += nodes;

                        node.childlren.push(n);

                        if self.end_of_search(env) {
                            break;
                        } else if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
                            is_timeout = true;
                            break;
                        }
                    },
                    RootEvaluationResult::Timeout(n) => {
                        node.childlren.push(n);

                        is_timeout = true;
                        break;
                    }
                }

                let event_queue = Arc::clone(&env.event_queue);
                event_dispatcher.dispatch_events(self,&*event_queue)?;

                if self.end_of_search(env) {
                    break;
                } else if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
                    is_timeout = true;
                    break;
                }
            } else if let Some(mut game_node) = node.childlren.pop() {
                match self.startup_strategy(env,
                                            gs,
                                            game_node.mhash,
                                            game_node.shash,
                                            game_node.m,
                                            is_oute) {
                    Some((oute_kyokumen_map, current_kyokumen_map)) => {
                        let next = env.kyokumen_map.get((gs.teban,game_node.mhash,game_node.shash)).map(|(_,state,mc)| {
                            (Arc::clone(state),Arc::clone(mc))
                        }).unwrap();

                        match next {
                            (state, mc) => {
                                let teban = gs.teban;
                                let current_depth = gs.current_depth;

                                let mut env = env.clone();
                                let evalutor = evalutor.clone();

                                let sender = sender.clone();

                                let b = std::thread::Builder::new();

                                let sender = sender.clone();

                                evalutor.on_begin_thread();

                                let _ = b.stack_size(1024 * 1024 * 200).spawn(move || {
                                    let mut event_dispatcher = Self::create_event_dispatcher::<Recursive<L, S>>(&env.on_error_handler, &env.stop, &env.quited);

                                    let mut gs = GameState {
                                        teban: teban.opposite(),
                                        state: &state,
                                        m: Some(m),
                                        mc: &mc,
                                        obtained: obtained,
                                        current_kyokumen_map: &current_kyokumen_map,
                                        oute_kyokumen_map: &oute_kyokumen_map,
                                        mhash: mhash,
                                        shash: shash,
                                        depth: depth - 1,
                                        current_depth: current_depth + 1
                                    };

                                    let strategy = Recursive::new();

                                    let r = strategy.search(&mut env,
                                                                                           &mut gs,
                                                                                           &mut game_node,
                                                                                    &mut event_dispatcher, &evalutor);

                                    let r = match r {
                                        Ok(EvaluationResult::Value(win,nodes,mvs)) => {
                                            Ok(RootEvaluationResult::Value(game_node,win, nodes, mvs))
                                        },
                                        Ok(EvaluationResult::Timeout) => {
                                            Ok(RootEvaluationResult::Timeout(game_node))
                                        },
                                        Err(e) => Err(e)
                                    };

                                    let _ = sender.send(r);
                                });

                                threads -= 1;
                            }
                        }
                    },
                    None => (),
                }
            } else if evalutor.active_threads() > 0 {
                threads -= 1;
            } else {
                break;
            }
        }

        self.termination(env, &mut gs, node, evalutor,scoreval)?;

        node.childlren.pop().ok_or(ApplicationError::InvalidStateError(String::from(
            "Node list is empty"
        ))).map(|n| {
            let mvs = match n {
                EvaluationResult::Value(_, _, mut mvs) => {
                    mvs.push_front(gs.m);
                    mvs
                },
                _ => VecDeque::new()
            };
            EvaluationResult::Value(Score::Value(0.),0,mvs)
        })
    }
}
impl<L,S> Search<L,S> for Root<L,S> where L: Logger + Send + 'static, S: InfoSender {
    fn search<'a,'b>(&self,env:&mut Environment<L,S>,
                     gs:&mut GameState<'a>,
                     node:&mut GameNode,
                     event_dispatcher:&mut UserEventDispatcher<'b,Root<L,S>,ApplicationError,L>,
                     evalutor: &Evalutor) -> Result<EvaluationResult,ApplicationError> {
        let r = self.parallelized(env,gs,node,event_dispatcher,evalutor);

        env.stop.store(true,atomic::Ordering::Release);

        match r {
            Ok(r) => {
                while evalutor.active_threads() > 0 {
                    self.receiver.recv()?.map_err(|e| ApplicationError::from(e))?;
                    evalutor.on_end_thread()?;
                }

                Ok(r)
            },
            Err(e) => {
                Err(e)
            }
        }
    }
}
pub struct Recursive<L,S> where L: Logger + Send + 'static, S: InfoSender {
    l:PhantomData<L>,
    s:PhantomData<S>,
}
impl<L,S> Recursive<L,S> where L: Logger + Send + 'static, S: InfoSender {
    pub fn new() -> Recursive<L,S> {
        Recursive {
            l:PhantomData::<L>,
            s:PhantomData::<S>,
        }
    }
}
impl<L,S> Search<L,S> for Recursive<L,S> where L: Logger + Send + 'static, S: InfoSender {
    fn search<'a,'b>(&self,env:&mut Environment<L,S>,
                     gs:&mut GameState<'a>,
                     node: &mut GameNode,
                     event_dispatcher:&mut UserEventDispatcher<'b,Recursive<L,S>,ApplicationError,L>,
                     evalutor: &Evalutor) -> Result<EvaluationResult,ApplicationError> {
        let mut gs = gs;

        let await_mvs = match self.before_search(env,&mut gs,event_dispatcher,evalutor)? {
            BeforeSearchResult::Complete(r) => {
                return Ok(r);
            },
            BeforeSearchResult::Terminate(None) => {
                return Ok(EvaluationResult::Value(Score::Value(0.),0,mvs));
            },
            BeforeSearchResult::Terminate(Some(s)) => {
                return Ok(EvaluationResult::Value(s,1,mvs));
            },
            BeforeSearchResult::AsyncMvs(mvs) => {
                mvs
            }
        };

        if await_mvs.len() > 0 {
            evalutor.begin_transaction()?;
        }

        for r in await_mvs {
            let (m,s) = r.0.recv()?;
            env.nodes.fetch_add(1,atomic::Ordering::Release);

            if let Some(mut g) = env.kyokumen_map.get_mut((gs.teban,mhash,shash)) {
                let (ref mut score,_,_) = *g;

                *score = s;
            }
        }

        for (mut gamenode) in node.childlren.peek_mut() {
            match startup_strategy(env,
                                   gs,
                                   game_node.mhash,
                                   game_node.shash,
                                   game_node.m,
                                   is_oute) {
                Some((oute_kyokumen_map,current_kyokumen_map)) => {
                    let next = env.kyokumen_map.get((gs.teban,game_node.mhash,game_node.shash)).map(|(_,state,mc)| {
                        (Arc::clone(state),Arc::clone(mc))
                    }).unwrap();

                    match next {
                        (state, mc) => {
                            let mut gs = GameState {
                                teban: gs.teban.opposite(),
                                state: &state,
                                m: Some(m),
                                mc: &mc,
                                obtained: obtained,
                                current_kyokumen_map: &current_kyokumen_map,
                                oute_kyokumen_map: &oute_kyokumen_map,
                                mhash: mhash,
                                shash: shash,
                                depth: depth - 1,
                                current_depth: gs.current_depth + 1
                            };

                            let strategy = Recursive::new();

                            match strategy.search(env, &mut gs, &mut gamenode, event_dispatcher,evalutor)? {
                                EvaluationResult::Timeout => {
                                    return Ok(EvaluationResult::Timeout);
                                },
                                RootEvaluationResult::Value(mut n, win, nodes,  mut mvs) => {
                                    n.win += win;
                                    n.nodes += nodes;

                                    node.childlren.push(n);

                                    mvs.push_front(m);

                                    return Ok(EvaluationResult::Value(-win,nodes,mvs));
                                }
                            }
                        }
                    }
                }
                None => (),
            }
        }

        Ok(EvaluationResult::Value(Score::Value(0.),0, best_moves))
    }
}