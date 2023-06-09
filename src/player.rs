use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{Ordering};
use std::time::{Duration, Instant};
use usiagent::command::{BestMove, CheckMate, UsiInfoSubCommand, UsiOptType};
use usiagent::error::{PlayerError, UsiProtocolError};
use usiagent::event::{GameEndState, SysEventOption, SysEventOptionKind, UserEvent, UserEventQueue, UsiGoMateTimeLimit, UsiGoTimeLimit};
use usiagent::hash::{KyokumenHash, KyokumenMap};
use usiagent::logger::{Logger};
use usiagent::OnErrorHandler;
use usiagent::output::USIOutputWriter;
use usiagent::player::{InfoSender, OnKeepAlive, PeriodicallyInfo, USIPlayer};
use usiagent::rule::{AppliedMove, Kyokumen, Rule, State};
use usiagent::shogi::{Banmen, Mochigoma, MochigomaCollections, Move, Teban};
use crate::error::{ApplicationError};
use crate::nn::Evalutor;
use crate::search::{Environment, GameNode, GameState, MAX_THREADS, MIN_TURN_COUNT, NETWORK_DELAY, Root, RootEvaluationResult, Score, Search, TURN_COUNT, TURN_LIMIT};

pub trait FromOption {
    fn from_option(option:SysEventOption) -> Option<Self> where Self: Sized;
}
impl FromOption for i64 {
    fn from_option(option: SysEventOption) -> Option<i64> {
        match option {
            SysEventOption::Num(v) => Some(v),
            _ => None
        }
    }
}
impl FromOption for u32 {
    fn from_option(option: SysEventOption) -> Option<u32> {
        match option {
            SysEventOption::Num(v) => Some(v as u32),
            _ => None
        }
    }
}
impl FromOption for usize {
    fn from_option(option: SysEventOption) -> Option<usize> {
        match option {
            SysEventOption::Num(v) => Some(v as usize),
            _ => None
        }
    }
}
impl FromOption for bool {
    fn from_option(option: SysEventOption) -> Option<bool> {
        match option {
            SysEventOption::Bool(b) => Some(b),
            _ => None
        }
    }
}
pub struct Raiden {
    savedir: String,
    nna_path:String,
    nnb_path:String,
    evalutor: Option<Evalutor>,
    kyokumen:Option<Kyokumen>,
    remaining_turns:u32,
    mhash:u64,
    shash:u64,
    oute_kyokumen_map:KyokumenMap<u64,()>,
    kyokumen_map:KyokumenMap<u64,u32>,
    pub history:Vec<(Banmen,MochigomaCollections,u64,u64)>,
    hasher:Arc<KyokumenHash<u64>>,
    turn_limit:Option<u32>,
    max_nodes:Option<i64>,
    max_threads:u32,
    network_delay:u32,
    turn_count:u32,
    min_turn_count:u32,
}
impl fmt::Debug for Raiden {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Leo")
    }
}
impl Raiden {
    pub fn new(savedir: String,
               nna_path:String,
               nnb_path:String) -> Raiden {
        Raiden {
            savedir:savedir,
            nna_path:nna_path,
            nnb_path:nnb_path,
            evalutor:None,
            kyokumen:None,
            remaining_turns:0,
            mhash:0,
            shash:0,
            oute_kyokumen_map:KyokumenMap::new(),
            kyokumen_map:KyokumenMap::new(),
            history:Vec::new(),
            hasher:Arc::new(KyokumenHash::new()),
            turn_limit:None,
            max_nodes:None,
            max_threads:MAX_THREADS,
            network_delay:NETWORK_DELAY,
            turn_count:TURN_COUNT,
            min_turn_count:MIN_TURN_COUNT,
        }
    }

    fn send_message_immediate<L,S>(&self, env:&mut Environment<L,S>, message:&str) -> Result<(),ApplicationError>
        where  L: Logger + Send + 'static,
               S: InfoSender,
               Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
        let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
        commands.push(UsiInfoSubCommand::Str(String::from(message)));

        Ok(env.info_sender.send_immediate(commands)?)
    }
}
impl USIPlayer<ApplicationError> for Raiden {
    const ID: &'static str = "raiden";
    const AUTHOR: &'static str = "j6k1";
    fn get_option_kinds(&mut self) -> Result<BTreeMap<String,SysEventOptionKind>,ApplicationError> {
        let mut kinds:BTreeMap<String,SysEventOptionKind> = BTreeMap::new();
        kinds.insert(String::from("USI_Hash"),SysEventOptionKind::Num);
        kinds.insert(String::from("USI_Ponder"),SysEventOptionKind::Bool);
        kinds.insert(String::from("MaxNodes"),SysEventOptionKind::Num);
        kinds.insert(String::from("TURN_COUNT"),SysEventOptionKind::Num);
        kinds.insert(String::from("MIN_TURN_COUNT"),SysEventOptionKind::Num);
        kinds.insert(String::from("Threads"),SysEventOptionKind::Num);
        kinds.insert(String::from("TurnLimit"),SysEventOptionKind::Num);
        kinds.insert(String::from("NetworkDelay"),SysEventOptionKind::Num);

        Ok(kinds)
    }
    fn get_options(&mut self) -> Result<BTreeMap<String,UsiOptType>,ApplicationError> {
        let mut options:BTreeMap<String,UsiOptType> = BTreeMap::new();
        options.insert(String::from("TurnLimit"),UsiOptType::Spin(1,3600000,Some(TURN_LIMIT as i64)));
        options.insert(String::from("MaxNodes"),UsiOptType::Spin(0,i64::MAX,Some(0)));
        options.insert(String::from("TURN_COUNT"),UsiOptType::Spin(0,1000,Some(TURN_COUNT as i64)));
        options.insert(String::from("MIN_TURN_COUNT"),UsiOptType::Spin(0,1000,Some(MIN_TURN_COUNT as i64)));
        options.insert(String::from("Threads"),UsiOptType::Spin(1,1024,Some(MAX_THREADS as i64)));
        options.insert(String::from("NetworkDelay"),UsiOptType::Spin(0,10000,Some(NETWORK_DELAY as i64)));

        Ok(options)
    }
    fn take_ready<W,L>(&mut self, _:OnKeepAlive<W,L>)
                       -> Result<(),ApplicationError> where W: USIOutputWriter + Send + 'static,
                                                       L: Logger + Send + 'static {
        match self.evalutor {
            Some(_) => (),
            None => {
                self.evalutor = Some(Evalutor::new(self.savedir.clone(),
                                              self.nna_path.clone(),
                                              self.nnb_path.clone())?);
            }
        }
        Ok(())
    }
    fn set_option(&mut self,name:String,value:SysEventOption) -> Result<(),ApplicationError> {
        match &*name {
            "TurnLimit" => {
                self.turn_limit = u32::from_option(value);
            },
            "MaxNodes" => {
                self.max_nodes = match i64::from_option(value) {
                    Some(0) => {
                        None
                    },
                    Some(nodes) => {
                        Some(nodes)
                    },
                    None => None,
                };
            },
            "Threads" => {
                self.max_threads = u32::from_option(value).unwrap_or(MAX_THREADS);
            },
            "NetworkDelay" => {
                self.network_delay = u32::from_option(value).unwrap_or(NETWORK_DELAY);
            },
            "TURN_COUNT" => {
                self.turn_count = u32::from_option(value).unwrap_or(TURN_COUNT);
            },
            "MIN_TURN_COUNT" => {
                self.min_turn_count = u32::from_option(value).unwrap_or(MIN_TURN_COUNT);
            },
            _ => ()
        }

        Ok(())
    }
    fn newgame(&mut self) -> Result<(),ApplicationError> {
        self.kyokumen = None;
        self.history.clear();
        self.remaining_turns = self.turn_count;
        Ok(())
    }
    fn set_position(&mut self,teban:Teban,banmen:Banmen,
                    ms:Mochigoma,mg:Mochigoma,_:u32,m:Vec<Move>)
                    -> Result<(),ApplicationError> {
        self.history.clear();
        self.kyokumen_map = KyokumenMap::new();

        let kyokumen_map:KyokumenMap<u64,u32> = KyokumenMap::new();
        let (mhash,shash) = self.hasher.calc_initial_hash(&banmen, &ms, &mg);

        let teban = teban;
        let state = State::new(banmen);

        let mc = MochigomaCollections::new(ms,mg);


        let history:Vec<(Banmen,MochigomaCollections,u64,u64)> = Vec::new();

        let (t,state,mc,r) = self.apply_moves(state,teban, mc,&m.into_iter()
                .map(|m| m.to_applied_move())
                .collect::<Vec<AppliedMove>>(),
        (mhash,shash,kyokumen_map,history),
          |_,t,banmen,mc,m,o,r| {
          let (prev_mhash,prev_shash,mut kyokumen_map,mut history) = r;

          let (mhash,shash) = match m {
              &Some(m) => {
                  let mhash = self.hasher.calc_main_hash(prev_mhash, t, &banmen, &mc, m, &o);
                  let shash = self.hasher.calc_sub_hash(prev_shash, t, &banmen, &mc, m, &o);

                  match kyokumen_map.get(t,&mhash,&shash).unwrap_or(&0) {
                      &c => {
                          kyokumen_map.insert(t,mhash,shash,c+1);
                      }
                  };
                  (mhash,shash)
              },
              &None => {
                  (prev_mhash,prev_shash)
              }
          };

          history.push((banmen.clone(),mc.clone(),prev_mhash,prev_shash));
          (mhash,shash,kyokumen_map,history)
        });

        let (mhash,shash,kyokumen_map,history) = r;

        let mut oute_kyokumen_map:KyokumenMap<u64,()> = KyokumenMap::new();
        let mut current_teban = t.opposite();

        let mut current_cont = true;
        let mut opponent_cont = true;

        for h in history.iter().rev() {
            match &h {
                &(ref banmen,_, mhash,shash) => {
                    if current_cont && Rule::is_mate(current_teban,&State::new(banmen.clone())) {
                        oute_kyokumen_map.insert(current_teban,*mhash,*shash,());
                    } else if !opponent_cont {
                        break;
                    } else {
                        current_cont = false;
                    }
                }
            }

            std::mem::swap(&mut current_cont, &mut opponent_cont);

            current_teban = current_teban.opposite();
        }

        self.kyokumen = Some(Kyokumen {
            state:state,
            mc:mc,
            teban:t
        });
        self.mhash = mhash;
        self.shash = shash;
        self.oute_kyokumen_map = oute_kyokumen_map;
        self.kyokumen_map = kyokumen_map;
        self.history = history;
        Ok(())
    }
    fn think<L,S,P>(&mut self,think_start_time:Instant,
                    limit:&UsiGoTimeLimit,event_queue:Arc<Mutex<UserEventQueue>>,
                    info_sender:S,periodically_info:P,on_error_handler:Arc<Mutex<OnErrorHandler<L>>>)
                    -> Result<BestMove,ApplicationError>
        where L: Logger + Send + 'static,
              S: InfoSender,
              P: PeriodicallyInfo {
        let (teban,state,mc) = self.kyokumen.as_ref().map(|k| (k.teban,&k.state,&k.mc)).ok_or(
            UsiProtocolError::InvalidState(
                String::from("Position information is not initialized."))
        )?;

        let limit = limit.to_instant(teban,think_start_time);
        let current_limit = limit.map(|l| think_start_time + (l  - think_start_time) / self.remaining_turns);

        let mut env = Environment::new(
            Arc::clone(&event_queue),
            info_sender.clone(),
            Arc::clone(&on_error_handler),
            Arc::clone(&self.hasher),
            think_start_time.clone(),
            limit,
            current_limit,
            self.turn_limit.map(|l| think_start_time + Duration::from_millis(l as u64)),
            self.turn_count,
            self.min_turn_count,
            self.max_nodes.clone(),
            self.network_delay,
            self.max_threads
        );

        let (mhash,shash) = (self.mhash.clone(), self.shash.clone());
        let kyokumen_map = self.kyokumen_map.clone();
        let oute_kyokumen_map = self.oute_kyokumen_map.clone();

        match self.evalutor {
            Some(ref evalutor) => {
                let mut event_dispatcher = Root::<L,S>::create_event_dispatcher(&on_error_handler,&env.stop,&env.quited);

                let _pinfo_sender = {
                    let nodes = env.nodes.clone();
                    let think_start_time = think_start_time.clone();
                    let on_error_handler = env.on_error_handler.clone();

                    periodically_info.start(100,move || {
                        let mut commands = vec![];
                        commands.push(UsiInfoSubCommand::Nodes(nodes.load(Ordering::Acquire)));

                        let sec = (Instant::now() - think_start_time).as_secs();

                        if sec > 0 {
                            commands.push(UsiInfoSubCommand::Nps(nodes.load(Ordering::Acquire) / sec));
                        }

                        commands
                    }, &on_error_handler)
                };

                let mut gs = GameState {
                    teban: teban,
                    state: &Arc::new(state.clone()),
                    m:None,
                    mc: &Arc::new(mc.clone()),
                    obtained:None,
                    current_kyokumen_map:&kyokumen_map,
                    oute_kyokumen_map:&oute_kyokumen_map,
                    mhash:mhash,
                    shash:shash,
                    current_depth:0
                };

                let strategy  = Root::new();

                let result = strategy.search(&mut env,&mut gs, &mut GameNode::new(None,mhash,shash,0), &mut event_dispatcher, evalutor);

                let bestmove = match result {
                    Err(ref e) => {
                        let _ = env.on_error_handler.lock().map(|h| h.call(e));
                        self.send_message_immediate(&mut env,format!("{}",e).as_str())?;
                        BestMove::Resign
                    },
                    Ok(RootEvaluationResult::Timeout) => {
                        self.send_message_immediate(&mut env,"think timeout!")?;
                        BestMove::Resign
                    },
                    Ok(RootEvaluationResult::Value(Score::NEGINFINITE, _, _)) => {
                        BestMove::Resign
                    },
                    Ok(RootEvaluationResult::Value(_, _, mvs)) if mvs.len() == 0 => {
                        self.send_message_immediate(&mut env,"moves is empty!")?;
                        BestMove::Resign
                    },
                    Ok(RootEvaluationResult::Value(_, _, mvs)) => {
                        BestMove::Move(mvs[0].to_move(),None)
                    }
                };

                if self.remaining_turns > env.min_turn_count {
                    self.remaining_turns -= 1;
                }

                Ok(bestmove)
            },
            None =>  {
                Err(ApplicationError::InvalidStateError(format!("evalutor is not initialized!")))
            }
        }
    }
    fn think_ponder<L,S,P>(&mut self,_:&UsiGoTimeLimit,_:Arc<Mutex<UserEventQueue>>,
                           _:S,_:P,_:Arc<Mutex<OnErrorHandler<L>>>)
                           -> Result<BestMove,ApplicationError> where L: Logger + Send + 'static, S: InfoSender,
                                                                 P: PeriodicallyInfo + Send + 'static {
        unimplemented!();
    }

    fn think_mate<L,S,P>(&mut self,_:&UsiGoMateTimeLimit,_:Arc<Mutex<UserEventQueue>>,
                         _:S,_:P,_:Arc<Mutex<OnErrorHandler<L>>>)
                         -> Result<CheckMate,ApplicationError>
        where L: Logger + Send + 'static,
              S: InfoSender,
              P: PeriodicallyInfo {
        unimplemented!();
    }
    
    fn on_stop(&mut self,_:&UserEvent) -> Result<(), ApplicationError> where ApplicationError: PlayerError {
        Ok(())
    }

    fn gameover<L>(&mut self,_:&GameEndState,
                   _:Arc<Mutex<UserEventQueue>>, _:Arc<Mutex<OnErrorHandler<L>>>) -> Result<(),ApplicationError> where L: Logger, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
        Ok(())
    }

    fn on_ponderhit(&mut self,_:&UserEvent) -> Result<(), ApplicationError> where ApplicationError: PlayerError {
        Ok(())
    }

    fn on_quit(&mut self,_:&UserEvent) -> Result<(), ApplicationError> where ApplicationError: PlayerError {
        Ok(())
    }

    fn quit(&mut self) -> Result<(),ApplicationError> {
        Ok(())
    }
}
