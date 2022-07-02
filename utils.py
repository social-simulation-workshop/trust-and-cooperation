import itertools
import numpy as np
import sys


def draw(p) -> bool:
    return True if np.random.uniform() < p else False

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class Agent:
    _ids = itertools.count(0)

    def __init__(self) -> None:
        self.id = next(self._ids)
        self.gene = Agent.generate_random_gene()
        self.iter = 0
        self.payoff = list()

        self.sum_of_weights = 0
        self.sum_of_weighted_payoff = 0
        self.fitness = 0
    

    @staticmethod
    def generate_random_gene() -> str:
        """ Return generated random genes. Only gene[1..15] are used."""
        return "".join([f"{b:0>8b}" for b in np.random.bytes(2)])[:16]
    
    
    @staticmethod
    def flip_gene(bit) -> str:
        return "1" if bit == "0" else "0"
    

    def get_gene(self, allele) -> True:
        return True if self.gene[allele] == "1" else False
    

    def display_marker(self) -> bool:
        if self.get_gene(1) and self.get_gene(2):
            return True
        if not self.get_gene(1) and not self.get_gene(2):
            return True
        return False
    

    def greet(self) -> bool:
        return True if self.get_gene(3) else False
    

    def coop(self, is_nei) -> bool:
        if self.get_gene(1):
            return True
        elif is_nei and self.get_gene(15):
            return True
        else:
            return False

    
    def is_paired(self, iter_idx) -> bool:
        return True if self.iter == iter_idx else False
    

    def mark_paired(self, iter_idx) -> None:
        self.iter = iter_idx
        self.payoff = list()
    

    def to_trust_ag(self, ag_b, is_nei) -> bool:
        trust = 0
        untrust = 0

        # gene 4
        if self.get_gene(4):
            if self.get_gene(9):
                # project (same as yours)
                if self.get_gene(1):
                    trust += 1
                else:
                    untrust += 1
            else:
                # negetive project
                if self.get_gene(1):
                    untrust += 1
                else:
                    trust += 1

        # gene 5
        if self.get_gene(5):
            if self.get_gene(10):
                # trust those who display marker
                if ag_b.display_marker():
                    trust += 1
                else:
                    untrust += 1
            else:
                # untrust those who display marker
                if ag_b.display_marker():
                    untrust += 1
                else:
                    trust += 1
        
        # gene 6
        if self.get_gene(6):
            if self.get_gene(11):
                # trust those who greet
                if ag_b.greet():
                    trust += 1
                else:
                    untrust += 1
            else:
                # untrust those who greet
                if ag_b.greet():
                    untrust += 1
                else:
                    trust += 1
        
        # gene 7
        if self.get_gene(7):
            if self.get_gene(12):
                # trust neighbors
                if is_nei:
                    trust += 1
                else:
                    untrust += 1
            else:
                # untrust neighbors
                if is_nei:
                    untrust += 1
                else:
                    trust += 1
        
        # gene 8
        if self.get_gene(8):
            if self.get_gene(13):
                # trust those who are relatively successful
                if self.fitness < ag_b.fitness:
                    trust += 1
                else:
                    untrust += 1
            else:
                if self.fitness < ag_b.fitness:
                    untrust += 1
                else:
                    trust += 1
        
        if trust + untrust != 0:
            return draw(trust/(trust+untrust))
        else:
            # attend to none of the 5 cues
            return self.get_gene(14)
    

    def receive_payoff(self, payoff):
        self.payoff.append(payoff)
    

    def update_fitness(self):
        """ Update fitness at the end of each iteration. """
        avg_payoff = sum(self.payoff) / len(self.payoff)
        self.sum_of_weighted_payoff = self.sum_of_weighted_payoff * 0.999 + avg_payoff
        self.sum_of_weights += 0.999 ** (self.iter - 1)
        self.fitness = self.sum_of_weighted_payoff / self.sum_of_weights                


class Network:
    PAYOFF_T = 4
    PAYOFF_R = 3
    PAYOFF_P = 1
    PAYOFF_S = 0

    def __init__(self, n_player=1000, n_neighbor=10, n_iteration=100000, embeddedness=2/3, payoff_x=1,
                 copy_rate=0.5, error_rate=0.01, verbose=True, rnd_seed=np.random.randint(10000)) -> None:
        np.random.seed(rnd_seed)
        self.rnd_seed = rnd_seed
        self.verbose = verbose

        self.n_player = n_player
        self.n_neighbor = n_neighbor
        self.n_neighborhood = n_player // n_neighbor
        self.n_iteration = n_iteration
        self.embeddedness = embeddedness
        self.copy_rate = copy_rate
        self.error_rate = error_rate
        self.PAYOFF_X = payoff_x

        self.ags = [Agent() for _ in range(n_player)]

        # record
        self.n_interact_nei = 0
        self.n_trust_in_nei = 0
        self.n_interact_stranger = 0
        self.n_trust_in_stranger = 0

        self.n_exchange_nei = 0
        self.n_coop_with_nei = 0
        self.n_exchange_stranger = 0
        self.n_coop_with_stranger = 0

        # record for plot
        self.g1_list = list()
        self.g15_list = list()
        self.trust_btw_nei = list()
        self.trust_btw_stranger = list()


    def propagate_strategy(self, ag_a: Agent, ag_b: Agent):
        if ag_a.fitness == ag_b.fitness:
            return
        
        # let ag_a be the less fit one
        if ag_a.fitness > ag_b.fitness:
            ag_a, ag_b = ag_b, ag_a
        
        ag_a_gene = list(ag_a.gene)
        for bit in range(1, 16):
            if draw(self.copy_rate):
                ag_a_gene[bit] = ag_b.gene[bit]
                if draw(self.error_rate):
                    ag_a_gene[bit] = Agent.flip_gene(ag_b.gene[bit])
                else:
                    ag_a_gene[bit] = ag_b.gene[bit]
        ag_a.gene = "".join(ag_a_gene)
            

    def _count_trust(self, trust, is_nei):
        if is_nei:
            self.n_interact_nei += 1
            if trust:
                self.n_trust_in_nei += 1
        else:
            self.n_interact_stranger += 1
            if trust:
                self.n_trust_in_stranger += 1
    

    def _count_coop(self, coop, is_nei):
        if is_nei:
            self.n_exchange_nei += 1
            if coop:
                self.n_coop_with_nei += 1
        else:
            self.n_exchange_stranger += 1
            if coop:
                self.n_coop_with_stranger += 1


    def interact(self, ag_a: Agent, ag_b: Agent, is_nei, iter_idx) -> None:
        # if self.verbose:
        #     print("interact | ag {} and ag {}, is_nei = {}".format(ag_a.id, ag_b.id, is_nei))
        #     print("paired:")
        #     print(" ".join(["{:2d}".format(i) for i in range(self.n_player)]))
        #     print(" ".join(["{:2d}".format(1 if ag.is_paired(iter_idx) else 0) for ag in self.ags]))
        
        ag_a_trust = ag_a.to_trust_ag(ag_b, is_nei)
        ag_b_trust = ag_b.to_trust_ag(ag_a, is_nei)
        self._count_trust(ag_a_trust, is_nei)
        self._count_trust(ag_b_trust, is_nei)

        if ag_a_trust and ag_b_trust:
            # exchange
            self._count_coop(ag_a.coop(is_nei), is_nei)
            self._count_coop(ag_b.coop(is_nei), is_nei)

            if ag_a.coop(is_nei) and ag_b.coop(is_nei):
                ag_a.receive_payoff(Network.PAYOFF_R)
                ag_b.receive_payoff(Network.PAYOFF_R)
            if not ag_a.coop(is_nei) and ag_b.coop(is_nei):
                ag_a.receive_payoff(Network.PAYOFF_T)
                ag_b.receive_payoff(Network.PAYOFF_S)
            if ag_a.coop(is_nei) and not ag_b.coop(is_nei):
                ag_a.receive_payoff(Network.PAYOFF_S)
                ag_b.receive_payoff(Network.PAYOFF_T)
            if not ag_a.coop(is_nei) and not ag_b.coop(is_nei):
                ag_a.receive_payoff(Network.PAYOFF_P)
                ag_b.receive_payoff(Network.PAYOFF_P)
        
        else:
            # either one exits
            ag_a.receive_payoff(self.PAYOFF_X)
            ag_b.receive_payoff(self.PAYOFF_X)
        
        # strategy propagation
        self.propagate_strategy(ag_a, ag_b)
    

    def record(self):
        self.g1_list.append(len([1 for ag in self.ags if ag.get_gene(1)])/self.n_player)
        self.g15_list.append(len([1 for ag in self.ags if ag.get_gene(15)])/self.n_player)
        self.trust_btw_nei.append(self.n_trust_in_nei / self.n_interact_nei)
        self.trust_btw_stranger.append(self.n_trust_in_stranger / self.n_interact_stranger)
        if self.verbose:
            self.print_result(to_stderr=True)

    
    def get_result(self) -> list:
        res = list()
        res.append(self.g1_list[-1])
        res.append(self.g15_list[-1])
        res.append(self.trust_btw_nei[-1])
        res.append(self.trust_btw_stranger[-1])
        res.append(self.n_coop_with_nei / self.n_exchange_nei)
        res.append(self.n_coop_with_stranger / self.n_exchange_stranger)
        return res
    

    @staticmethod
    def _print_result_formatter(res: list, stddev=None, to_stderr=True):
        assert len(res) == 6
        if stddev is not None:
            print("coop          (Gene 1) : {:.4f} ({:.4f})".format(res[0], stddev[0]))
            print("coop with nei (Gene 15): {:.4f} ({:.4f})".format(res[1], stddev[1]))
            print("trust in nei           : {:.4f} ({:.4f})".format(res[2], stddev[2]))
            print("trust in stranger      : {:.4f} ({:.4f})".format(res[3], stddev[3]))
            print("coop with nei          : {:.4f} ({:.4f})".format(res[4], stddev[4]))
            print("coop with stranger     : {:.4f} ({:.4f})".format(res[5], stddev[5]))
            print()
        elif to_stderr:
            eprint("coop          (Gene 1) :", res[0])
            eprint("coop with nei (Gene 15):", res[1])
            eprint("trust in nei           :", res[2])
            eprint("trust in stranger      :", res[3])
            eprint("coop with nei          :", res[4])
            eprint("coop with stranger     :", res[5])
            eprint()
        else:
            print("coop          (Gene 1) :", res[0])
            print("coop with nei (Gene 15):", res[1])
            print("trust in nei           :", res[2])
            print("trust in stranger      :", res[3])
            print("coop with nei          :", res[4])
            print("coop with stranger     :", res[5])
            print()
    

    def print_result(self, to_stderr=True):
        self._print_result_formatter(self.get_result(), to_stderr=to_stderr)
    

    @staticmethod
    def print_multi_run_result(res: list):
        res = np.array(res)
        mean = np.mean(res, axis=0)
        stddev = np.std(res, axis=0)
        print("==== RESULTS OF MUTLIPLE REPLICATIONS ====")
        Network._print_result_formatter(list(mean), list(stddev))
    

    def get_suffix_str(self) -> str:
        return "_".join(["player_{}".format(self.n_player),
                          "neighbor_{}".format(self.n_neighbor),
                          "embeddedness_{}".format(self.embeddedness),
                          "iter_{}".format(self.n_iteration),
                          "payoffx_{}".format(self.PAYOFF_X),
                          "rndseed_{}".format(self.rnd_seed)])

    
    def simulate(self, log_verbose_n=10, log_record_v=200):
        eprint("params: n_player={}, n_neighbor={}, n_iteration={}, embeddedness={}, payoff_x={}, rnd_seed={}".format(self.n_player,
            self.n_neighbor, self.n_iteration, self.embeddedness, self.PAYOFF_X, self.rnd_seed))
        log_idx, log_t_list = 0, [int(self.n_iteration*((i+1)/log_verbose_n)) for i in range(log_verbose_n)]
        for iter_idx in range(1, self.n_iteration+1):
            if self.verbose:
                if log_t_list[log_idx] == iter_idx:
                    eprint("t: {:6d}/{:6d} ({:.1f}%)".format(iter_idx, self.n_iteration,
                        100*(iter_idx)/self.n_iteration))
                    log_idx += 1
            
            for ag_a_id in range(self.n_player):
                if self.ags[ag_a_id].is_paired(iter_idx):
                    continue
                
                if draw(self.embeddedness):
                    # inside the neighborhood
                    ag_a_neihd_id = ag_a_id // self.n_neighbor
                    ag_a_nei_id = ag_a_id % self.n_neighbor
                    ag_b_nei_id = (ag_a_nei_id + np.random.randint(1, self.n_neighbor)) % self.n_neighbor
                    ag_b_id = ag_a_neihd_id * self.n_neighbor + ag_b_nei_id
                    is_nei = True
                else:
                    # outside the neighborhood
                    ag_a_neihd_id = ag_a_id // self.n_neighbor
                    ag_b_neihd_id = (ag_a_neihd_id + np.random.randint(1, self.n_neighborhood)) % self.n_neighborhood
                    ag_b_nei_id = np.random.randint(self.n_neighbor)
                    ag_b_id = ag_b_neihd_id * self.n_neighbor + ag_b_nei_id
                    is_nei = False
                
                if not self.ags[ag_a_id].is_paired(iter_idx):
                    self.ags[ag_a_id].mark_paired(iter_idx)
                if not self.ags[ag_b_id].is_paired(iter_idx):
                    self.ags[ag_b_id].mark_paired(iter_idx)
                self.interact(self.ags[ag_a_id], self.ags[ag_b_id], is_nei, iter_idx)
            
            for ag in self.ags:
                ag.update_fitness()
            
            if iter_idx % log_record_v == 0:
                self.record()
        
        # print("params: n_player={}, n_neighbor={}, n_iteration={}, embeddedness={}, payoff_x={}, rnd_seed={}".format(self.n_player,
        #     self.n_neighbor, self.n_iteration, self.embeddedness, self.PAYOFF_X, self.rnd_seed))
        # self.print_result(to_stderr=True)

