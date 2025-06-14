import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ----- Scheduling Algorithms -----
def fcfs(processes):
    # Sort by arrival
    plist = sorted(processes, key=lambda x: x['arrival'])
    time = 0
    schedule = []
    for p in plist:
        start = max(time, p['arrival'])
        end = start + p['burst']
        schedule.append({'pid': p['pid'], 'start': start, 'end': end})
        time = end
    return schedule


def sjf_nonpreemptive(processes):
    procs = processes.copy()
    time = 0
    schedule = []
    while procs:
        ready = [p for p in procs if p['arrival'] <= time]
        if not ready:
            time += 1
            continue
        p = min(ready, key=lambda x: x['burst'])
        start = time
        end = time + p['burst']
        schedule.append({'pid': p['pid'], 'start': start, 'end': end})
        time = end
        procs.remove(p)
    return schedule


def sjf_preemptive(processes):
    procs = [p.copy() for p in processes]
    time = 0
    schedule = []
    while any(p['burst'] > 0 for p in procs):
        ready = [p for p in procs if p['arrival'] <= time and p['burst'] > 0]
        if not ready:
            time += 1
            continue
        p = min(ready, key=lambda x: x['burst'])
        # run one time unit
        schedule.append({'pid': p['pid'], 'start': time, 'end': time+1})
        p['burst'] -= 1
        time += 1
    # merge contiguous
    merged = []
    for seg in schedule:
        if merged and merged[-1]['pid']==seg['pid'] and merged[-1]['end']==seg['start']:
            merged[-1]['end'] = seg['end']
        else:
            merged.append(seg.copy())
    return merged


def round_robin(processes, quantum):
    procs = sorted([p.copy() for p in processes], key=lambda x: x['arrival'])
    queue = []
    time = 0
    schedule = []
    i = 0
    while i < len(procs) or queue:
        # enqueue arrived
        while i < len(procs) and procs[i]['arrival'] <= time:
            queue.append(procs[i])
            i += 1
        if not queue:
            time += 1
            continue
        p = queue.pop(0)
        run = min(quantum, p['burst'])
        schedule.append({'pid': p['pid'], 'start': time, 'end': time+run})
        p['burst'] -= run
        time += run
        # enqueue newly arrived during run
        while i < len(procs) and procs[i]['arrival'] <= time:
            queue.append(procs[i]); i+=1
        if p['burst'] > 0:
            queue.append(p)
    return schedule

# ----- Metrics -----
def compute_metrics(schedule, processes):
    # completion times
    comp = {}
    for seg in schedule:
        comp[seg['pid']] = seg['end']
    tot_turn = sum(comp[p['pid']] - p['arrival'] for p in processes)
    tot_wait = sum((comp[p['pid']] - p['arrival'] - p['burst_orig']) for p in processes)
    # response: first start
    first = {}
    for seg in schedule:
        pid = seg['pid']
        if pid not in first:
            first[pid] = seg['start']
    tot_resp = sum(first[p['pid']] - p['arrival'] for p in processes)
    n = len(processes)
    return {
        'avg_turnaround': tot_turn/n,
        'avg_waiting': tot_wait/n,
        'avg_response': tot_resp/n
    }

# ----- Streamlit UI -----
st.title("Process Scheduling Simulator")

alg = st.selectbox("Choose Algorithm", ["FCFS", "SJF (Non-preemptive)", "SJF (Preemptive)", "Round Robin"] )
num = st.number_input("Number of Processes", min_value=1, max_value=10, value=3, step=1)

processes = []
cols = st.columns(3)
for i in range(int(num)):
    with cols[0]: pid = st.number_input(f"PID {i+1}", value=i+1, key=f"pid_{i}")
    with cols[1]: arriv = st.number_input(f"Arrival {i+1}", min_value=0, value=0, key=f"arr_{i}")
    with cols[2]: burst = st.number_input(f"Burst {i+1}", min_value=1, value=1, key=f"bur_{i}")
    processes.append({'pid': pid, 'arrival': arriv, 'burst': burst, 'burst_orig': burst})

quantum = None
if alg.startswith("Round Robin"):
    quantum = st.number_input("Time Quantum", min_value=1, value=2)

if st.button("Simulate"):
    # run selected
    if alg == "FCFS":
        schedule = fcfs(processes)
    elif alg == "SJF (Non-preemptive)":
        schedule = sjf_nonpreemptive(processes)
    elif alg == "SJF (Preemptive)":
        schedule = sjf_preemptive(processes)
    else:
        schedule = round_robin(processes, quantum)

    # metrics
    metrics = compute_metrics(schedule, processes)
    st.metric("Avg Turnaround Time", f"{metrics['avg_turnaround']:.2f}")
    st.metric("Avg Waiting Time", f"{metrics['avg_waiting']:.2f}")
    st.metric("Avg Response Time", f"{metrics['avg_response']:.2f}")

    # Gantt chart
    fig, ax = plt.subplots(figsize=(8, 3))
    for seg in schedule:
        ax.add_patch(Rectangle((seg['start'], seg['pid']-0.4), seg['end']-seg['start'], 0.8))
        ax.text(seg['start'] + (seg['end']-seg['start'])/2, seg['pid'], str(seg['pid']),
                va='center', ha='center')
    ax.set_ylim(0.5, max(p['pid'] for p in processes)+0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Process ID')
    st.pyplot(fig)
