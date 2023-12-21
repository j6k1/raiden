use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::marker::PhantomData;
use std::ops::{Add, Deref, Neg};
use std::sync::{Arc, atomic, mpsc, Mutex, RwLock, TryLockError, Weak};
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
use crate::initial_estimation::{attack_priority, defense_priority};
use crate::nn::Evalutor;

pub const TURN_LIMIT:u32 = 1000;
pub const TIMELIMIT_MARGIN:u64 = 50;
pub const NETWORK_DELAY:u32 = 1100;
pub const MAX_THREADS:u32 = 1;
pub const TURN_COUNT:u32 = 50;
pub const MIN_TURN_COUNT:u32 = 5;

pub trait Search<L,S>: Sized where L: Logger + Send + 'static, S: InfoSender {
    type Output;

    fn search<'a,'b>(&self,env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                     node: &Weak<RwLock<GameNode>>,
                     event_dispatcher:&mut UserEventDispatcher<'b,Self,ApplicationError,L>,
                     evalutor: &Evalutor) -> Result<Self::Output,ApplicationError>;

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

        env.info_sender.send(commands)?;
        Ok(env.info_sender.flush()?)
    }

    fn send_message_immediate(&self, env:&mut Environment<L,S>, message:&str) -> Result<(),ApplicationError>
        where Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
        let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
        commands.push(UsiInfoSubCommand::Str(String::from(message)));

        Ok(env.info_sender.send_immediate(commands)?)
    }

    fn send_depth(&self, env:&mut Environment<L,S>, depth:u32) -> Result<(),ApplicationError> {
        let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
        commands.push(UsiInfoSubCommand::Depth(depth));

        Ok(env.info_sender.send(commands)?)
    }

    fn send_info(&self, env:&mut Environment<L,S>, pv:&VecDeque<LegalMove>, score:Score) -> Result<(),ApplicationError>
        where Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
        let depth = pv.len() as u32;

        let mut commands: Vec<UsiInfoSubCommand> = Vec::new();

        match score {
            Score::INFINITE => {
                commands.push(UsiInfoSubCommand::Score(UsiScore::Mate(UsiScoreMate::Plus)));
            },
            Score::NEGINFINITE => {
                commands.push(UsiInfoSubCommand::Score(UsiScore::Mate(UsiScoreMate::Minus)));
            },
            Score::Value(s) => {
                commands.push(UsiInfoSubCommand::Score(UsiScore::Cp((s * (1 << 23) as f32) as  i64)));
            }
        }

        commands.push(UsiInfoSubCommand::Depth(depth));

        if pv.len() > 0 {
            commands.push(UsiInfoSubCommand::CurrMove(pv[0].to_move()));
            commands.push(UsiInfoSubCommand::Pv(pv.clone().into_iter().map(|m| m.to_move()).collect()));
        }
        commands.push(UsiInfoSubCommand::Time((Instant::now() - env.think_start_time).as_millis() as u64));

        env.info_sender.send(commands)?;

        Ok(env.info_sender.flush()?)
    }

    fn startup_strategy<'a>(&self,gs: &mut GameState<'a>,
                            mhash:u64,
                            shash:u64,
                            m:LegalMove,
                            is_oute:bool)
                            -> (Option<ObtainKind>,KyokumenMap<u64,()>,KyokumenMap<u64,u32>,u32) {
        let obtained = match m {
            LegalMove::To(ref m) => m.obtained(),
            _ => None,
        };

        let mut oute_kyokumen_map = gs.oute_kyokumen_map.clone();
        let mut current_kyokumen_map = gs.current_kyokumen_map.clone();

        if is_oute {
            match oute_kyokumen_map.get(gs.teban,&mhash,&shash) {
                None => {
                    oute_kyokumen_map.insert(gs.teban,mhash,shash,());
                },
                _ => ()
            }
        }

        if !is_oute {
            oute_kyokumen_map.clear(gs.teban);
        }

        let sennichite_count = match current_kyokumen_map.get(gs.teban,&mhash,&shash).unwrap_or(&0) {
            &c if c > 0 => {
                current_kyokumen_map.insert(gs.teban,mhash,shash,c+1);

                c
            },
            _ => 0,
        };

        (obtained,oute_kyokumen_map,current_kyokumen_map,sennichite_count)
    }

    fn before_search<'a,'b>(&self,
                         env: &mut Environment<L, S>,
                         gs: &mut GameState<'a>,
                         node:&Weak<RwLock<GameNode>>,
                         raw_node:&'b mut GameNode,
                         evalutor: &Evalutor)
        -> Result<BeforeSearchResult, ApplicationError> {
        if raw_node.expanded {
            return Ok(BeforeSearchResult::Recur);
        }

        if self.end_of_search(env) {
            return Ok(BeforeSearchResult::Terminate(None));
        } else if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
            return Ok(BeforeSearchResult::Timeout);
        }

        if let Some(ObtainKind::Ou) = gs.obtained {
            return Ok(BeforeSearchResult::Terminate(Some(Score::INFINITE)));
        }

        let mvs = Rule::win_only_moves(gs.teban,&gs.state);

        if mvs.len() > 0 {
            let m = mvs[0];

            let mut mvs = VecDeque::new();

            mvs.push_front(m);

            return Ok(BeforeSearchResult::Complete(mvs,Score::INFINITE, 1));
        }

        let (mvs,defense) = if Rule::is_mate(gs.teban.opposite(),&*gs.state) {
            let mvs = Rule::respond_oute_only_moves_all(gs.teban, &*gs.state, &*gs.mc);

            if mvs.len() == 0 {
                return Ok(BeforeSearchResult::Complete(VecDeque::new(),Score::NEGINFINITE, 1));
            } else {
                (mvs,true)
            }
        } else {
            let mvs:Vec<LegalMove> = Rule::legal_moves_all(gs.teban, &*gs.state, &*gs.mc);

            if mvs.len() == 0 {
                return Ok(BeforeSearchResult::Complete(VecDeque::new(),Score::NEGINFINITE,1));
            } else {
                (mvs,false)
            }
        };

        if self.end_of_search(env) {
            return Ok(BeforeSearchResult::Terminate(None));
        } else if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
            return Ok(BeforeSearchResult::Timeout);
        }

        for &m in &mvs {
            let o = match m {
                LegalMove::To(ref m) => {
                    m.obtained().and_then(|o| MochigomaKind::try_from(o).ok())
                },
                _ => None
            };

            let mhash = env.hasher.calc_main_hash(gs.mhash,gs.teban,gs.state.get_banmen(),&gs.mc,m.to_applied_move(),&o);
            let shash = env.hasher.calc_sub_hash(gs.shash,gs.teban,gs.state.get_banmen(),&gs.mc,m.to_applied_move(),&o);

            let n = if gs.current_depth == 0 && defense {
                Arc::new(RwLock::new(GameNode::new(Weak::new(),Some(m), mhash, shash, defense_priority(gs.teban,&gs.state,m))))
            } else if gs.current_depth == 0 {
                Arc::new(RwLock::new(GameNode::new(Weak::new(),Some(m), mhash, shash, attack_priority(gs.teban,&gs.state,m))))
            } else if defense {
                Arc::new(RwLock::new(GameNode::new(node.clone(),Some(m), mhash, shash, defense_priority(gs.teban,&gs.state,m))))
            } else {
                Arc::new(RwLock::new(GameNode::new(node.clone(),Some(m), mhash, shash, attack_priority(gs.teban,&gs.state,m))))
            };

            if !env.kyokumen_map.contains_key(&(gs.teban.opposite(),mhash,shash)) {
                env.kyokumen_map.insert_new((gs.teban.opposite(), mhash, shash), (Arc::clone(&gs.state),Arc::clone(&gs.mc)));
            }

            env.nodes.fetch_add(1,atomic::Ordering::Release);

            let (state,mc,_) = Rule::apply_move_none_check(&gs.state, gs.teban, &gs.mc, m.to_applied_move());

            evalutor.submit(gs.teban.opposite(),state.get_banmen(),&mc,Arc::clone(&n))?;

            raw_node.childlren.push(WrappedGameNode { node: n });
        }

        Ok(BeforeSearchResult::AsyncMvs)
    }
}
#[derive(Debug)]
pub enum EvaluationResult {
    Done,
    Pending,
    Timeout,
}
#[derive(Debug)]
pub enum RootEvaluationResult {
    Value(Score,u64,VecDeque<LegalMove>),
    Timeout,
}
#[derive(Debug)]
pub enum RecvEvaluationResult {
    Done(WrappedGameNode),
    Pending(WrappedGameNode),
    Timeout(WrappedGameNode),
}
#[derive(Debug)]
pub enum BeforeSearchResult {
    Complete(VecDeque<LegalMove>,Score,u64),
    Timeout,
    Terminate(Option<Score>),
    AsyncMvs,
    Recur
}
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub enum Score {
    NEGINFINITE,
    Value(f32),
    INFINITE
}
impl Neg for Score {
    type Output = Score;

    fn neg(self) -> Score {
        match self {
            Score::NEGINFINITE => Score::INFINITE,
            Score::Value(v) => Score::Value(-v),
            Score::INFINITE => Score::NEGINFINITE
        }
    }
}
impl Add for Score {
    type Output = Self;

    fn add(self, other:Score) -> Self::Output {
        match (self,other) {
            (Score::INFINITE,_) => Score::INFINITE,
            (Score::Value(l),Score::Value(r)) => Score::Value(l + r),
            (Score::Value(_),Score::NEGINFINITE) => Score::NEGINFINITE,
            (Score::Value(_),Score::INFINITE) => Score::INFINITE,
            (Score::NEGINFINITE,_) => Score::NEGINFINITE
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
    pub max_nodes:Option<i64>,
    pub network_delay:u32,
    pub max_threads:u32,
    pub stop:Arc<AtomicBool>,
    pub quited:Arc<AtomicBool>,
    pub kyokumen_map:Arc<ConcurrentFixedHashMap<(Teban,u64,u64),(Arc<State>,Arc<MochigomaCollections>)>>,
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
            max_nodes:self.max_nodes.clone(),
            network_delay:self.network_delay,
            max_threads:self.max_threads,
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
               max_nodes:Option<i64>,
               network_delay:u32,
               max_threads:u32,
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
            max_nodes:max_nodes,
            network_delay:network_delay,
            max_threads:max_threads,
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
    pub current_depth:u32
}
#[derive(Debug)]
pub struct WrappedGameNode {
    node:Arc<RwLock<GameNode>>
}
impl Clone for WrappedGameNode {
    fn clone(&self) -> Self {
        WrappedGameNode { node: Arc::clone(&self.node) }
    }
}
impl Deref for WrappedGameNode {
    type Target = Arc<RwLock<GameNode>>;

    fn deref(&self) -> &Self::Target {
        &self.node
    }
}

impl PartialEq<Self> for WrappedGameNode {
    fn eq(&self, other: &Self) -> bool {
        self.node.read().unwrap().eq(&other.read().unwrap())
    }
}
impl Eq for WrappedGameNode {

}
impl PartialOrd for WrappedGameNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.node.read().unwrap().partial_cmp(&other.node.read().unwrap())
    }
}
impl Ord for WrappedGameNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.node.read().unwrap().cmp(&other.node.read().unwrap())
    }
}
#[derive(Debug)]
pub struct GameNode {
    pub win:Score,
    pub nodes:u64,
    expanded:bool,
    asc_priority:i32,
    pub mate_nodes:usize,
    m:Option<LegalMove>,
    mhash:u64,
    shash:u64,
    pub parent:Weak<RwLock<GameNode>>,
    pub childlren:BinaryHeap<WrappedGameNode>
}
impl GameNode {
    pub fn new(parent:Weak<RwLock<GameNode>>,m:Option<LegalMove>,mhash:u64,shash:u64,asc_priority:i32) -> GameNode {
        GameNode {
            win:Score::Value(0.),
            nodes:0,
            expanded:false,
            asc_priority:asc_priority,
            mate_nodes:0,
            m:m,
            mhash:mhash,
            shash:shash,
            parent:parent,
            childlren:BinaryHeap::new()
        }
    }

    pub fn computed_score(&self) -> Score {
        if !self.expanded {
            Score::NEGINFINITE
        } else {
            match self.win {
                Score::INFINITE => Score::INFINITE,
                Score::NEGINFINITE => Score::NEGINFINITE,
                Score::Value(win) => {
                    Score::Value(win / self.nodes as f32)
                }
            }
        }
    }

    pub fn best_moves(&mut self) -> Result<VecDeque<LegalMove>,ApplicationError> {
        let mut nodes = VecDeque::new();
        let mut parents = VecDeque::new();

        let mut mvs = VecDeque::new();

        let c = self.childlren.pop();

        let mut c = if let Some(ref n) = &c {
            n.read().map(|n| {
                n.m.map(|m| mvs.push_back(m));
            })?;

            nodes.push_front(Arc::clone(&n.node));
            parents.push_front(Arc::clone(&n.node));

            n.clone()
        } else {
            return Ok(mvs);
        };

        loop {
            let n = c.write()?.childlren.pop();

            if let Some(ref n) = &n {
                n.read().map(|n| {
                    n.m.map(|m| mvs.push_back(m));
                })?;

                let n = WrappedGameNode { node: Arc::clone(&n.node) };

                nodes.push_front(Arc::clone(&n.node));
                parents.push_front(Arc::clone(&n.node));

                if n.read().map(|n| n.expanded)? {
                    c = n;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if let Some(n) = nodes.pop_back() {
            self.childlren.push(WrappedGameNode { node: n });
        }

        for (p,n) in parents.into_iter().skip(1).zip(nodes.into_iter()) {
            match p.write() {
                Ok(mut p) => {
                    p.childlren.push(WrappedGameNode { node:n });
                },
                Err(e) => {
                    return Err(e)?;
                }
            }
        }

        self.m.map(|m| mvs.push_front(m));

        Ok(mvs)
    }

    pub fn best_score(&mut self) -> Result<Score,ApplicationError> {
        let mut nodes = VecDeque::new();

        loop {
            let n = if let Some(ref n) = self.childlren.pop() {
                n.node.clone()
            } else {
                break;
            };

            nodes.push_front(Arc::clone(&n));

            match n.read() {
                Ok(n) => {
                    if n.expanded {
                        let s = n.computed_score();

                        for n in nodes {
                            self.childlren.push(WrappedGameNode { node: n });
                        }

                        return Ok(-s);
                    }
                },
                Err(e) => {
                    return Err(e)?;
                }
            };
        }

        Ok(Score::NEGINFINITE)
    }
}
impl Ord for GameNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl PartialOrd for GameNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let l = self.computed_score();
        let r = other.computed_score();

        if l == Score::NEGINFINITE && r == Score::NEGINFINITE {
           Some(self.expanded.cmp(&other.expanded)
                          .then(self.asc_priority.cmp(&other.asc_priority))
                          .reverse().then((self as *const GameNode).cmp(&(other as *const GameNode))))
        } else {
            l.partial_cmp(&r).map(|r| {
                r.reverse().then((self as *const GameNode).cmp(&(other as *const GameNode)))
            })
        }
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
    receiver:Receiver<Result<RecvEvaluationResult, ApplicationError>>,
    sender:Sender<Result<RecvEvaluationResult, ApplicationError>>
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

    fn parallelized<'a,'b>(&self,env:&mut Environment<L,S>,
                           gs:&mut GameState<'a>,
                           node:&Weak<RwLock<GameNode>>,
                           event_dispatcher:&mut UserEventDispatcher<'b,Root<L,S>,ApplicationError,L>,
                           evalutor: &Evalutor) -> Result<RootEvaluationResult,ApplicationError> {
        let mut gs = gs;

        let current_node = node.upgrade();

        let mut raw_node = match current_node.as_ref() {
            Some(n) => {
                match n.try_write() {
                    Err(TryLockError::WouldBlock) => {
                        return Err(ApplicationError::LogicError(String::from("root node lock failed.")));
                    },
                    Ok(n) => {
                        n
                    },
                    Err(TryLockError::Poisoned(e)) => {
                        return Err(ApplicationError::from(e));
                    }
                }
            },
            None => {
                return Err(ApplicationError::LogicError(String::from(
                    "current node is empty."
                )));
            }
        };

        match self.before_search(env, &mut gs, node, &mut raw_node, evalutor)? {
            BeforeSearchResult::Complete(mvs,win,nodes) => {
                return Ok(RootEvaluationResult::Value(win, nodes, mvs));
            },
            BeforeSearchResult::Timeout | BeforeSearchResult::Terminate(None) => {
                return Ok(RootEvaluationResult::Timeout);
            },
            BeforeSearchResult::Terminate(Some(win)) => {
                raw_node.win = win;
                raw_node.expanded = true;
                raw_node.nodes = 1;

                return Ok(RootEvaluationResult::Value(win, 1, VecDeque::new()));
            },
            BeforeSearchResult::AsyncMvs => {
                evalutor.on_begin_thread()?;
                evalutor.begin_transaction()?;
            },
            BeforeSearchResult::Recur => {}
        }

        let mvs_count = raw_node.childlren.len();
        let mut best_score = Score::NEGINFINITE;

        let mut threads = env.max_threads.min(mvs_count as u32);

        let sender = self.sender.clone();

        let mut is_timeout = false;

        let mut completed = false;

        loop {
            let is_mate = raw_node.childlren.peek().map(|n| {
                n.read().map(|n| n.expanded && n.computed_score() == Score::INFINITE)
            }).unwrap_or(Ok(false))?;

            if (completed || is_timeout || is_mate) && evalutor.active_threads() == 0 {
                break;
            }

            if threads == 0 || completed || is_timeout || is_mate {
                let r = self.receiver.recv();

                evalutor.on_end_thread()?;

                let r = r?.map_err(|e| ApplicationError::from(e))?;

                threads += 1;

                match r {
                    RecvEvaluationResult::Done(n) | RecvEvaluationResult::Pending(n) => {
                        let (win,expanded,computed_score) = match n.write() {
                            Ok(mut n) => {
                                if n.expanded && best_score <= -n.computed_score() {
                                    best_score = -n.computed_score();

                                    let pv = n.best_moves()?;

                                    self.send_info(env, &pv, -n.computed_score())?;
                                }

                                let win = if n.expanded && n.computed_score() == Score::INFINITE {
                                    raw_node.mate_nodes += 1;

                                    if raw_node.mate_nodes == mvs_count {
                                        Some(Score::INFINITE)
                                    } else {
                                        Some(Score::Value(1.))
                                    }
                                } else {
                                    None
                                };

                                (win,n.expanded, n.computed_score())
                            },
                            Err(e) => {
                                return Err(ApplicationError::from(e))
                            }
                        };

                        if expanded && (win == Some(Score::INFINITE) || computed_score == Score::NEGINFINITE) {
                            raw_node.childlren.push(n);
                            completed = true;
                            env.stop.store(true, atomic::Ordering::Release);
                        } else {
                            raw_node.childlren.push(n);
                        }

                        if self.end_of_search(env) {
                            completed = true;
                            env.stop.store(true, atomic::Ordering::Release);
                        } else if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
                            is_timeout = true;
                            env.stop.store(true, atomic::Ordering::Release);
                        }
                    },
                    RecvEvaluationResult::Timeout(n) => {
                        raw_node.childlren.push(n);
                        is_timeout = true;
                        env.stop.store(true, atomic::Ordering::Release);
                    }
                }

                let event_queue = Arc::clone(&env.event_queue);
                event_dispatcher.dispatch_events(self, &*event_queue)?;

                if self.end_of_search(env) {
                    completed = true;
                    env.stop.store(true, atomic::Ordering::Release);
                } else if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
                    is_timeout = true;
                    env.stop.store(true, atomic::Ordering::Release);
                }
            } else if let Some(game_node) = raw_node.childlren.pop() {
                let mut next_node = game_node.write()?;

                let m = next_node.m.ok_or(ApplicationError::InvalidStateError(
                    String::from(
                        "Move is None."
                    )
                ))?;

                match self.startup_strategy(gs,
                                            next_node.mhash,
                                            next_node.shash,
                                            m,
                                            Rule::is_oute_move(&gs.state,gs.teban, m)) {
                    (obtained,
                        oute_kyokumen_map,
                        current_kyokumen_map,
                        sennichite_count) => {

                        if sennichite_count >= 3 && !next_node.expanded {
                            next_node.expanded = true;
                            next_node.nodes = 1;
                            next_node.win = Score::INFINITE;

                            raw_node.childlren.push(game_node.clone());

                            raw_node.mate_nodes += 1;

                            if raw_node.mate_nodes == mvs_count {
                                raw_node.win = Score::NEGINFINITE;
                                completed = true;
                                env.stop.store(true, atomic::Ordering::Release);
                            } else {
                                raw_node.win = raw_node.win + Score::Value(-1.);
                            }

                            raw_node.nodes += 1;

                            continue;
                        }

                        let next = env.kyokumen_map.get(&(gs.teban.opposite(), next_node.mhash, next_node.shash))
                            .map(|g| {
                                let (ref state, ref mc) = *g;
                                (Arc::clone(state), Arc::clone(mc))
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

                                evalutor.on_begin_thread()?;

                                let m = next_node.m;
                                let mhash = next_node.mhash;
                                let shash = next_node.shash;

                                drop(next_node);

                                let game_node = game_node.clone();

                                let _ = b.stack_size(1024 * 1024 * 200).spawn(move || {
                                    let game_node = game_node;
                                    let mut event_dispatcher = Self::create_event_dispatcher::<Recursive<L, S>>(&env.on_error_handler, &env.stop, &env.quited);

                                    let mut gs = GameState {
                                        teban: teban.opposite(),
                                        state: &state,
                                        m: m,
                                        mc: &mc,
                                        obtained: obtained,
                                        current_kyokumen_map: &current_kyokumen_map,
                                        oute_kyokumen_map: &oute_kyokumen_map,
                                        mhash: mhash,
                                        shash: shash,
                                        current_depth: current_depth + 1
                                    };

                                    let strategy = Recursive::new();

                                    let r = strategy.search(&mut env,
                                                            &mut gs,
                                                            &Arc::downgrade(&Arc::clone(&game_node.node)),
                                                            &mut event_dispatcher, &evalutor);

                                    let r = match r {
                                        Ok(EvaluationResult::Done) => {
                                            Ok(RecvEvaluationResult::Done(game_node.clone()))
                                        },
                                        Ok(EvaluationResult::Timeout) => {
                                            Ok(RecvEvaluationResult::Timeout(game_node.clone()))
                                        },
                                        Ok(EvaluationResult::Pending) => {
                                            Ok(RecvEvaluationResult::Pending(game_node.clone()))
                                        },
                                        Err(e) => Err(e)
                                    };

                                    let _ = sender.send(r);
                                });

                                threads -= 1;
                            }
                        }
                    }
                }
            } else {
                return Err(ApplicationError::InvalidStateError(String::from(
                    "The queue is empty."
                )));
            }
        }

        let best_score = raw_node.best_score()?;
        let best_moves = raw_node.best_moves()?;
        let nodes = raw_node.childlren.len() as u64;

        if is_timeout && best_moves.is_empty() {
            Ok(RootEvaluationResult::Timeout)
        } else {
            self.send_info(env, &best_moves, best_score)?;

            Ok(RootEvaluationResult::Value(best_score, nodes, best_moves))
        }
    }
}
impl<L,S> Search<L,S> for Root<L,S> where L: Logger + Send + 'static, S: InfoSender {
    type Output = RootEvaluationResult;

    fn search<'a,'b>(&self,env:&mut Environment<L,S>,
                     gs:&mut GameState<'a>,
                     node:&Weak<RwLock<GameNode>>,
                     event_dispatcher:&mut UserEventDispatcher<'b,Root<L,S>,ApplicationError,L>,
                     evalutor: &Evalutor) -> Result<Self::Output,ApplicationError> {
        let r = self.parallelized(env,gs,node,event_dispatcher,evalutor);

        env.stop.store(true,atomic::Ordering::Release);

        let mut last_err = None;

        match r {
            Ok(r) => {
                while evalutor.active_threads() > 0 {
                    if let Err(e) = self.receiver.recv()?.map_err(|e| ApplicationError::from(e)).and_then(|_| {
                        evalutor.on_end_thread()
                    }) {
                        last_err = Some(e);
                    }
                }

                match last_err {
                    None => Ok(r),
                    Some(e) => Err(e)
                }
            },
            Err(e) => {
                let mut last_err = Err(e);

                while evalutor.active_threads() > 0 {
                    if let Err(e) = self.receiver.recv()?.map_err(|e| ApplicationError::from(e)).and_then(|_| {
                        evalutor.on_end_thread()
                    }) {
                        last_err = Err(e);
                    }
                }

                last_err
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
    type Output = EvaluationResult;

    fn search<'a,'b>(&self,env:&mut Environment<L,S>,
                     gs:&mut GameState<'a>,
                     node: &Weak<RwLock<GameNode>>,
                     event_dispatcher:&mut UserEventDispatcher<'b,Recursive<L,S>,ApplicationError,L>,
                     evalutor: &Evalutor) -> Result<Self::Output,ApplicationError> {
        let mut gs = gs;

        let current_node = node.upgrade();

        let mut raw_node = match current_node.as_ref() {
            Some(n) => {
                match n.try_write() {
                    Err(TryLockError::WouldBlock) => {
                        return Ok(EvaluationResult::Pending);
                    },
                    Ok(n) => {
                        n
                    },
                    Err(TryLockError::Poisoned(e)) => {
                        return Err(ApplicationError::from(e));
                    }
                }
            },
            None => {
                return Err(ApplicationError::LogicError(String::from(
                    "current node is empty."
                )));
            }
        };

        if raw_node.expanded {
            let mvs_count = raw_node.childlren.len();

            let game_node = {
                let game_node_peek_mut = raw_node.childlren.peek_mut();

                let game_node_peek_mut = if let Some(game_node_peek_mut) = game_node_peek_mut {
                    game_node_peek_mut
                } else {
                    return Err(ApplicationError::LogicError(String::from(
                        "moves is empty."
                    )));
                };

                (*game_node_peek_mut).clone()
            };

            let mut next_node = match game_node.try_write() {
                Err(TryLockError::WouldBlock) => {
                    return Ok(EvaluationResult::Pending)
                },
                Ok(n) => {
                    n
                },
                Err(TryLockError::Poisoned(e)) => {
                    return Err(ApplicationError::from(e));
                }
            };

            let m = next_node.m.ok_or(ApplicationError::InvalidStateError(
                String::from(
                    "Move is None."
                )
            ))?;

            match self.startup_strategy(gs,
                                        next_node.mhash,
                                        next_node.shash,
                                        m,
                                        Rule::is_oute_move(&gs.state,gs.teban,m)) {
                (obtained,
                    oute_kyokumen_map,
                    current_kyokumen_map,
                    sennichite_count) => {
                    if sennichite_count >= 3 && !next_node.expanded {
                        next_node.expanded = true;
                        next_node.nodes = 1;
                        next_node.win = Score::INFINITE;

                        raw_node.mate_nodes += 1;

                        let win = if raw_node.mate_nodes == mvs_count {
                            Score::NEGINFINITE
                        } else {
                            raw_node.win + Score::Value(-1.)
                        };

                        raw_node.win = win;
                        raw_node.nodes += 1;

                        return Ok(EvaluationResult::Done);
                    }

                    let next = env.kyokumen_map.get(&(gs.teban.opposite(), next_node.mhash, next_node.shash))
                        .map(|g| {
                            let (ref state, ref mc) = *g;
                            (Arc::clone(state), Arc::clone(mc))
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
                                mhash: next_node.mhash,
                                shash: next_node.shash,
                                current_depth: gs.current_depth + 1
                            };

                            let strategy = Recursive::new();

                            drop(next_node);

                            match strategy.search(env, &mut gs, &Arc::downgrade(&game_node.node), event_dispatcher, evalutor)? {
                                EvaluationResult::Timeout => {
                                    return Ok(EvaluationResult::Timeout);
                                },
                                EvaluationResult::Pending => {
                                    return Ok(EvaluationResult::Pending);
                                },
                                EvaluationResult::Done => {
                                    return Ok(EvaluationResult::Done);
                                }
                            }
                        }
                    }
                }
            }
        } else {
            match self.before_search(env, &mut gs, node, &mut raw_node, evalutor)? {
                BeforeSearchResult::Complete(_, win, nodes) => {
                    raw_node.win = win;
                    raw_node.expanded = true;
                    raw_node.nodes = nodes;

                    self.send_depth(env,gs.current_depth)?;

                    Ok(EvaluationResult::Pending)
                },
                BeforeSearchResult::Timeout => {
                    self.send_depth(env,gs.current_depth)?;

                    Ok(EvaluationResult::Timeout)
                },
                BeforeSearchResult::Terminate(None) => {
                    raw_node.expanded = true;

                    self.send_depth(env,gs.current_depth)?;

                    Ok(EvaluationResult::Pending)
                },
                BeforeSearchResult::Terminate(Some(win)) => {
                    raw_node.win = win;
                    raw_node.expanded = true;
                    raw_node.nodes = 1;

                    self.send_depth(env,gs.current_depth)?;

                    Ok(EvaluationResult::Pending)
                },
                BeforeSearchResult::AsyncMvs => {
                    raw_node.expanded = true;

                    self.send_depth(env,gs.current_depth)?;
                    evalutor.begin_async_transaction()?;

                    Ok(EvaluationResult::Pending)
                },
                BeforeSearchResult::Recur => {
                    Err(ApplicationError::LogicError(String::from(
                        "The result of before_search is invalid"
                    )))
                }
            }
        }
    }
}