# RAN scheduler in Python
# Code for students that implements simulated RAN over time.
# Students need to code missing parts of main routine, as well as scheduling functions RR(), BET(), MT(), PF()
# Last update: Apr 19, 2024 - Nikko

# required numpy library
import csv
import numpy as np

# Constants
CQI_MAX = 15
tmax = 1000
N_UE = 10
N_RB = 12
MAX_LINE_LENGTH = 100
NUM_VALUES = 11
SYMBOLS_PER_RB = 84

# Global arrays
# CQI = np.zeros((tmax + 1, N_UE), dtype=int)
# arriving_bits = np.zeros((tmax + 1, N_UE), dtype=int)
BITS_PER_SYMBOL_by_CQI_value = np.array([
    0,
    2, 2, 2, 2,
    4, 4, 4, 4,
    6, 6, 6, 6,
    8, 8, 8
])
beta = 0.125
n_betas = 43  # number of betas for convergence to 0

# queue_length_in_bits = np.array([500] * N_UE)
# sum_queue_length_in_bits = np.zeros(N_UE, dtype=int)
# sum_nbits_sent_so_far = np.zeros(N_UE, dtype=int)


# INIT FUNCTION - load files
def init_load_files(CQI_FILENAME="data/PA1_CQI.dat", nbit_FILENAME="data/PA1_arriving_bits.dat"):
    CQI = np.zeros((tmax + 1, N_UE), dtype=int)
    arriving_bits = np.zeros((tmax + 1, N_UE), dtype=int)
    try:
        # Open and read in CQI values
        with open(CQI_FILENAME) as file:
            reader = csv.reader(file)
            for row in reader:
                tval = int(row[0])
                CQI[tval] = row[1:]

        # Open and read in number of bit arrivals per time unit per UE values
        with open(nbit_FILENAME) as file:
            reader = csv.reader(file)
            for row in reader:
                tval = int(row[0])
                arriving_bits[tval] = row[1:]

    except FileNotFoundError as e:
        print(f"Unable to open file: {e}")
        exit()

    except Exception as e:
        print(f"Error reading or processing the files: {e}")
        exit()

    return (CQI, arriving_bits)


def print_info_before(t, CQI_t, arriving_bits_t, queue_length_in_bits, sum_nbits_sent_so_far):
    print("Scheduler info at time:", t)
    print("  BEFORE scheduling:")
    print("    CQI for each UE queue:          ", " ".join(f"{x:06d}" for x in CQI_t))
    print("    bits/symbol per UE, for UE CQI: ", " ".join(f"{BITS_PER_SYMBOL_by_CQI_value[x]:06d}" for x in CQI_t))
    print("    nbits arriving, per UE queue:   ", " ".join(f"{x:06d}" for x in arriving_bits_t))
    print("    #bits in each UE queue:         ", " ".join(f"{x:06d}" for x in queue_length_in_bits))
    print("    nbits sent so far, per UE queue:", " ".join(f"{x:06d}" for x in sum_nbits_sent_so_far))


def print_info_after(UE_to_RB_count, UE_to_RB_nbits, queue_length_in_bits, sum_nbits_sent_so_far):
    print("  AFTER scheduling:")
    print("    # RB's assigned to each UE:     ", " ".join(f"{x:06d}" for x in list(UE_to_RB_count)))
    print("    #bits sent by each UE this time:", " ".join(f"{x:06d}" for x in list(UE_to_RB_nbits)))
    print("    #bits in each UE queue:         ", " ".join(f"{x:06d}" for x in queue_length_in_bits))
    print("    nbits sent so far, per UE queue:", " ".join(f"{x:06d}" for x in sum_nbits_sent_so_far))


def main():
    # Initialize the simulation 

    # load files
    CQI, arriving_bits = init_load_files()

    # current CQI value for UE[0] through UE[9]
    current_CQI = [0] * N_UE
    # current # bits arriving for eventual transmission for UE[0] through UE[9]
    current_nbits_arriving = [0] * N_UE
    # current bits in the queues
    queue_length_in_bits = np.array([500] * N_UE)
    # total bits sent 
    sum_nbits_sent_so_far = np.zeros(N_UE, dtype=int)
    # total queue length
    sum_queue_length_in_bits = np.zeros(N_UE, dtype=int)
    # performance stats averaged over all tmx time steps
    avg_queue_length_in_bits = [0] * N_UE
    avg_throughput = [0] * N_UE

    # track previous rates attained by each UE
    R = np.ones((1, N_UE))
    betas = np.zeros((n_betas, 1))
    for i in range(0, n_betas):
        betas[i] = np.power(beta, i+1)
    betas = np.flip(betas)

    # Simulation loop for each time step
    for t in range(1, tmax + 1):

        # get CQI values for UE's 0 ... 9 for this time unit into current_QCI[0] ... current_QCI[9]
        for i in range(0, N_UE):
            current_CQI[i] = CQI[t][i]

        # get number of arriving bits for each UE for this time unit E
        for i in range(0, N_UE):
            current_nbits_arriving[i] = arriving_bits[t][i]
    
        # udpate queue length of unsent bits for each UE by adding in number of current arrived bits
        # update sum of queue lengths over all intervals for each UE, so we can compute average queue length per UE later
        # initialize   UE_to_RB_count[] and UE_to_RB_nbits[], which will be completed by scheduler.
        for i in range(0, N_UE):
            queue_length_in_bits[i] += current_nbits_arriving[i]
            sum_queue_length_in_bits[i] += queue_length_in_bits[i]

        # UE_to_RB_count[i] in # of RBs assigned to UEi, returned by the scheduler
        UE_to_RB_count = np.zeros(N_UE, dtype=int)
        # UE_to_RB_count[i] in # bits from UEi queue to be tranmsitted in RBs this time unit returned by the scheduler
        UE_to_RB_nbits = np.zeros(N_UE, dtype=int)

        # Optional printing of detailed information before processing
        if t <= 6:
            print_info_before(t, CQI[t], arriving_bits[t], queue_length_in_bits, sum_nbits_sent_so_far)

        ########################################################################
        ## Scheduling: Students will implement RR, BET, MT, or PF scheduling. ##
        ########################################################################
        
        # change RR to other scheduling.
        # if RR(UE_to_RB_count, UE_to_RB_nbits, queue_length_in_bits, sum_nbits_sent_so_far, current_CQI) == -1:
        #     print("panic: RR scheduler returned -1")
        #     break

        rc = MT(UE_to_RB_count, UE_to_RB_nbits, queue_length_in_bits, sum_nbits_sent_so_far, current_CQI)
        if rc != 0:
            print(f"panic: MT scheduler returned {rc}")
            break
        #
        # rc, R = BET(UE_to_RB_count, UE_to_RB_nbits,
        #          queue_length_in_bits, sum_nbits_sent_so_far,
        #          R, betas, current_CQI)
        # if rc != 0:
        #     print(f"panic: BET scheduler returned {rc}")
        #     break
        #
        # rc, R = PF(UE_to_RB_count, UE_to_RB_nbits,
        #             queue_length_in_bits, sum_nbits_sent_so_far,
        #             R, betas, current_CQI)
        # if rc != 0:
        #     print(f"panic: PF scheduler returned {rc}")
        #     break

        for i in range(0, N_UE):
            # update variables
            queue_length_in_bits[i] -= UE_to_RB_nbits[i]
            sum_nbits_sent_so_far[i] += UE_to_RB_nbits[i]
        # print(UE_to_RB_nbits)
        # print("##")

        ########################################################################
        ## END HERE
        ########################################################################

        # Optional printing of detailed information after processing
        if t <= 6:
            print_info_after(UE_to_RB_count, UE_to_RB_nbits, queue_length_in_bits, sum_nbits_sent_so_far)

    # Compute average metrics after all time steps
    for i in range(0, N_UE):
        avg_queue_length_in_bits[i] = sum_queue_length_in_bits[i] // tmax
        avg_throughput[i] = sum_nbits_sent_so_far[i] // tmax

    # Printing final average metrics
    print("\n    Average throughput per UE:      ", ' '.join(f"{x:06d}" for x in avg_throughput))
    print("\n    Average UE queue length in bits:", ' '.join(f"{x:06d}" for x in avg_queue_length_in_bits))
    print()


########################################################
## Implement your RR, BET, MT, or PF scheduling here. ##
########################################################

# for RR - UE that is currently first/next in Round Robin sequence to get assigned an RB.  GLOBAL variable
RR_current_UE = 0


# implement round robin scheduling at time t, given queue_length_in_bits[i],CQI[t][i] for UEi
# update global vars to reflect  transmitted bits from queuelength count, increase # bits trasnmitted for each UE under RR
def RR(UE_to_RB_count, UE_to_RB_nbits, queue_length_in_bits, sum_nbits_sent_so_far, current_CQI): 

    global RR_current_UE

    # internal RR use, updated as RBs assigned; main() will update queue_length_in_bits[]
    RR_internal_queue_length_in_bits = np.copy(queue_length_in_bits)       

    # assign i_th RB to a UE
    for i in range(0, N_RB):
        nzerobits = 0
        # find next UE with data bits to be sent, starting at RR_current_UE
        # check RR-internal queue length variable to see if there are enough bits to make at least one symbol for RR_current_UE. 
        while RR_internal_queue_length_in_bits[RR_current_UE] < BITS_PER_SYMBOL_by_CQI_value[current_CQI[RR_current_UE]]:
            RR_current_UE = (RR_current_UE + 1) % N_UE
            nzerobits += 1
            # no data to send at all - if all queues empty there's a problem!
            if nzerobits > N_UE:
                return -1

        # get CQI for this UE
        current_UE_CQI = current_CQI[RR_current_UE]
        # get # bits per symbol for this UE, given CQI for this UE
        bits_per_symbol = BITS_PER_SYMBOL_by_CQI_value[current_UE_CQI]
        # current queue length for this UE in symbols
        nsymbols = RR_internal_queue_length_in_bits[RR_current_UE] // bits_per_symbol
        # if more than 84 symbols worth of data in queue, fill entire RB with 84 symbols
        nsymbols_sent = min(nsymbols, SYMBOLS_PER_RB)

        # total bits to sent
        nbits_sent = nsymbols_sent * bits_per_symbol

        # increase total bits sent to this UE
        UE_to_RB_nbits[RR_current_UE] += nbits_sent
        # increase total RB counter to this UE
        UE_to_RB_count[RR_current_UE] += 1
        
        # update internal QL
        RR_internal_queue_length_in_bits[RR_current_UE] -= nbits_sent;  

        # move to the next UE
        RR_current_UE = (RR_current_UE + 1) % N_UE

    return 0


# implement blind equal throughput (BET) scheduling at time t, given queue_length_in_bits[i],
# current_CQI[i] and any other variables or data structures you need to add
def BET(UE_to_RB_count, UE_to_RB_nbits, queue_length_in_bits, sum_nbits_sent_so_far, R, betas, current_CQI):
    betas_for_iter = betas
    R_for_iter = R
    if R.shape[0] < betas.shape[0]:
        betas_for_iter = betas[-R.shape[0]:, :]
    elif R.shape[0] == betas.shape[0]:
        pass  # no op
    elif R.shape[0] > betas.shape[0]:
        R_for_iter = R_for_iter[:betas.shape[0], :]
    prev_throughput = np.squeeze(np.matmul(np.transpose(betas_for_iter), R_for_iter))
    queue_length_for_iter = np.copy(queue_length_in_bits)
    for i in range(0, N_RB):
        min_throughput = np.finfo(np.float32).max
        min_throughput_ue_index = 0
        for j in range(0, N_UE):
            if prev_throughput[j] < min_throughput and queue_length_for_iter[j] > 0:
                min_throughput = prev_throughput[j]
                min_throughput_ue_index = j
        UE_to_RB_count[min_throughput_ue_index] += 1
        cqi_for_ue = current_CQI[min_throughput_ue_index]
        bits_per_symbol_for_ue = BITS_PER_SYMBOL_by_CQI_value[cqi_for_ue]
        bits_for_rb = min(
            queue_length_for_iter[min_throughput_ue_index],
            bits_per_symbol_for_ue * SYMBOLS_PER_RB
        )
        UE_to_RB_nbits[min_throughput_ue_index] += bits_for_rb
        queue_length_for_iter[min_throughput_ue_index] -= bits_for_rb
    R = np.vstack([R, np.expand_dims(np.copy(UE_to_RB_nbits), 0)])
    return 0, R


# implement maximum throughput (MT) scheduling at time t, given queue_length_in_bits[i], current_CQI[i]
# and any other variables or data structures you need to add
def MT(UE_to_RB_count, UE_to_RB_nbits, queue_length_in_bits, sum_nbits_sent_so_far, current_CQI):
    # Placeholder for the Maximum Throughput scheduling function
    queue_length_in_bits_internal = np.copy(queue_length_in_bits)
    for i in range(0, N_RB):
        MAX_BITS_PER_SYMBOL = 0
        MAX_RATE_UE_INDEX = -1
        for j in range(N_UE):
            CQI_FOR_UE = current_CQI[j]
            if CQI_FOR_UE > MAX_BITS_PER_SYMBOL and queue_length_in_bits_internal[j] > 0:
                MAX_BITS_PER_SYMBOL = CQI_FOR_UE
                MAX_RATE_UE_INDEX = j
            # BITS_PER_SYMBOL_FOR_UE = BITS_PER_SYMBOL_by_CQI_value[CQI_FOR_UE]
            # if BITS_PER_SYMBOL_FOR_UE > MAX_BITS_PER_SYMBOL and queue_length_in_bits_internal[j] > 0:
            #     MAX_BITS_PER_SYMBOL = BITS_PER_SYMBOL_FOR_UE
        UE_to_RB_count[MAX_RATE_UE_INDEX] += 1
        CQI_FOR_UE = current_CQI[MAX_RATE_UE_INDEX]
        BITS_PER_SYMBOL_FOR_UE = BITS_PER_SYMBOL_by_CQI_value[CQI_FOR_UE]
        BITS_FOR_RB = min(
            queue_length_in_bits_internal[MAX_RATE_UE_INDEX],
            BITS_PER_SYMBOL_FOR_UE * SYMBOLS_PER_RB
        )
        UE_to_RB_nbits[MAX_RATE_UE_INDEX] += BITS_FOR_RB
        queue_length_in_bits_internal[MAX_RATE_UE_INDEX] -= BITS_FOR_RB
    return 0


# implement proportionally fair (PF) scheduling at time t, given queue_length_in_bits[i],current_CQI[i] 
# and any other variables or data strcutures you need to add
def PF(UE_to_RB_count, UE_to_RB_nbits, queue_length_in_bits, sum_nbits_sent_so_far, R, betas, current_CQI):
    betas_for_iter = betas
    R_for_iter = R
    if R.shape[0] < betas.shape[0]:
        betas_for_iter = betas[-R.shape[0]:, :]
    elif R.shape[0] == betas.shape[0]:
        pass  # no op
    elif R.shape[0] > betas.shape[0]:
        R_for_iter = R_for_iter[:betas.shape[0], :]
    prev_throughput = np.squeeze(np.matmul(np.transpose(betas_for_iter), R_for_iter))
    queue_length_for_iter = np.copy(queue_length_in_bits)
    for i in range(0, N_RB):
        max_scheduler_value = np.finfo(np.float32).min
        max_scheduler_ue_index = 0
        for j in range(0, N_UE):
            CQI_FOR_UE = current_CQI[j]
            # BITS_PER_SYMBOL_FOR_UE = BITS_PER_SYMBOL_by_CQI_value[CQI_FOR_UE]
            # BITS_FOR_TIME_STEP = min(
            #     queue_length_for_iter[j],
            #     BITS_PER_SYMBOL_FOR_UE * SYMBOLS_PER_RB
            # )
            # BITS_FOR_TIME_STEP = BITS_PER_SYMBOL_FOR_UE * SYMBOLS_PER_RB
            if CQI_FOR_UE / prev_throughput[j] > max_scheduler_value and queue_length_for_iter[j] > 0:
                max_scheduler_value = CQI_FOR_UE / prev_throughput[j]
                max_scheduler_ue_index = j
            # if BITS_PER_SYMBOL_FOR_UE / prev_throughput[j] > max_scheduler_value and queue_length_for_iter[j] > 0:
            #     max_scheduler_value = BITS_PER_SYMBOL_FOR_UE / prev_throughput[j]
            #     max_scheduler_ue_index = j
        UE_to_RB_count[max_scheduler_ue_index] += 1
        cqi_for_ue = current_CQI[max_scheduler_ue_index]
        bits_per_symbol_for_ue = BITS_PER_SYMBOL_by_CQI_value[cqi_for_ue]
        bits_for_rb = min(
            queue_length_for_iter[max_scheduler_ue_index],
            bits_per_symbol_for_ue * SYMBOLS_PER_RB
        )
        UE_to_RB_nbits[max_scheduler_ue_index] += bits_for_rb
        queue_length_for_iter[max_scheduler_ue_index] -= bits_for_rb
    R = np.vstack([R, np.expand_dims(np.copy(UE_to_RB_nbits), 0)])
    return 0, R


if __name__ == '__main__':
    # start main function
    main()

