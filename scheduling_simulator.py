import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import heapq

# ----- Scheduling Algorithms -----
def first_come_first_served(process_list):
    sorted_list = sorted(process_list, key=lambda p: p['arrival'])
    current_time = 0
    timeline = []
    for process in sorted_list:
        start_time = max(current_time, process['arrival'])
        finish_time = start_time + process['burst']
        timeline.append({'process_id': process['process_id'], 'start_time': start_time, 'finish_time': finish_time})
        current_time = finish_time
    return timeline


def shortest_job_first_non_preemptive(process_list):
    processes = process_list.copy()
    current_time = 0
    timeline = []
    while processes:
        ready = [p for p in processes if p['arrival'] <= current_time]
        if not ready:
            current_time += 1
            continue
        selected = min(ready, key=lambda p: p['burst'])
        start_time = max(current_time, selected['arrival'])
        finish_time = start_time + selected['burst']
        timeline.append({'process_id': selected['process_id'], 'start_time': start_time, 'finish_time': finish_time})
        current_time = finish_time
        processes.remove(selected)
    return timeline


def shortest_job_first_preemptive(process_list):
    remaining = {p['process_id']: p['burst'] for p in process_list}
    processes = process_list.copy()
    current_time = 0
    timeline = []
    last_pid = None
    heap = []
    completed = 0
    total = len(process_list)
    while completed < total:
        for p in processes:
            if p['arrival'] == current_time:
                heapq.heappush(heap, (remaining[p['process_id']], p['arrival'], p['process_id'], p))
        if heap:
            burst, arrival, pid, p = heapq.heappop(heap)
            if pid != last_pid:
                timeline.append({'process_id': pid, 'start_time': current_time, 'finish_time': None})
            remaining[pid] -= 1
            current_time += 1
            last_pid = pid
            if remaining[pid] > 0:
                heapq.heappush(heap, (remaining[pid], arrival, pid, p))
            else:
                completed += 1
                timeline[-1]['finish_time'] = current_time
        else:
            current_time += 1
    # Merge and remove any segments without finish_time
    merged = []
    for seg in timeline:
        if seg['finish_time'] is None:
            continue
        if merged and merged[-1]['process_id'] == seg['process_id'] and merged[-1]['finish_time'] == seg['start_time']:
            merged[-1]['finish_time'] = seg['finish_time']
        else:
            merged.append(seg)
    return merged


def priority_non_preemptive(process_list):
    processes = process_list.copy()
    current_time = 0
    timeline = []
    while processes:
        ready = [p for p in processes if p['arrival'] <= current_time]
        if not ready:
            current_time += 1
            continue
        selected = min(ready, key=lambda p: (p['priority'], p['arrival']))
        start_time = max(current_time, selected['arrival'])
        finish_time = start_time + selected['burst']
        timeline.append({'process_id': selected['process_id'], 'start_time': start_time, 'finish_time': finish_time})
        current_time = finish_time
        processes.remove(selected)
    return timeline


def priority_preemptive(process_list):
    remaining = {p['process_id']: p['burst'] for p in process_list}
    processes = process_list.copy()
    current_time = 0
    timeline = []
    last_pid = None
    heap = []
    completed = 0
    total = len(process_list)
    while completed < total:
        for p in processes:
            if p['arrival'] == current_time:
                heapq.heappush(heap, (p['priority'], p['arrival'], p['process_id'], p))
        if heap:
            priority, arrival, pid, p = heapq.heappop(heap)
            if pid != last_pid:
                timeline.append({'process_id': pid, 'start_time': current_time, 'finish_time': None})
            remaining[pid] -= 1
            current_time += 1
            last_pid = pid
            if remaining[pid] > 0:
                heapq.heappush(heap, (priority, arrival, pid, p))
            else:
                completed += 1
                timeline[-1]['finish_time'] = current_time
        else:
            current_time += 1
    # Merge and remove incomplete
    merged = []
    for seg in timeline:
        if seg['finish_time'] is None:
            continue
        if merged and merged[-1]['process_id'] == seg['process_id'] and merged[-1]['finish_time'] == seg['start_time']:
            merged[-1]['finish_time'] = seg['finish_time']
        else:
            merged.append(seg)
    return merged


def round_robin_scheduling(process_list, quantum):
    queue = []
    timeline = []
    processes = sorted(process_list.copy(), key=lambda p: p['arrival'])
    current_time = 0
    idx = 0
    remaining = {p['process_id']: p['burst'] for p in processes}
    while idx < len(processes) or queue:
        while idx < len(processes) and processes[idx]['arrival'] <= current_time:
            queue.append(processes[idx])
            idx += 1
        if not queue:
            current_time += 1
            continue
        p = queue.pop(0)
        pid = p['process_id']
        start_time = current_time
        run_time = min(quantum, remaining[pid])
        current_time += run_time
        remaining[pid] -= run_time
        timeline.append({'process_id': pid, 'start_time': start_time, 'finish_time': current_time})
        while idx < len(processes) and processes[idx]['arrival'] <= current_time:
            queue.append(processes[idx])
            idx += 1
        if remaining[pid] > 0:
            queue.append(p)
    return timeline


def multi_level_queue_scheduling(process_list, queue_definitions):
    combined = []
    for level in sorted(queue_definitions):
        definition = queue_definitions[level]
        procs = [p for p in process_list if p['queue'] == level]
        if not procs:
            continue
        alg = definition['algorithm']
        if alg == 'FCFS':
            sched = first_come_first_served(procs)
        elif alg == 'RR':
            sched = round_robin_scheduling(procs, definition['quantum'])
        else:
            sched = priority_non_preemptive(procs)
        combined.extend(sched)
    return sorted(combined, key=lambda seg: seg['start_time'])


def compute_performance_metrics(timeline, process_list):
    # Validate unique IDs
    pids = [p['process_id'] for p in process_list]
    if len(pids) != len(set(pids)):
        st.error("Each process must have a unique Process ID.")
        st.stop()

    # Filter timeline
    timeline = [seg for seg in timeline if seg['finish_time'] is not None]
    completion = {seg['process_id']: seg['finish_time'] for seg in timeline}
    start_times = {}
    for seg in timeline:
        pid = seg['process_id']
        if pid not in start_times:
            start_times[pid] = seg['start_time']
    stats = []
    for p in process_list:
        pid = p['process_id']
        turnaround = completion[pid] - p['arrival']
        waiting = turnaround - p['burst']
        response = start_times[pid] - p['arrival']
        stats.append({'process_id': pid, 'turnaround': turnaround, 'waiting': waiting, 'response': response})
    df = pd.DataFrame(stats)
    return df.mean().to_dict(), df

# ----- Streamlit Interface -----
st.title("Process Scheduling Simulator")

selected = st.selectbox("Scheduling Algorithm", [
    "First Come First Serve",
    "Shortest Job First (Non-Preemptive)",
    "Shortest Job First (Preemptive)",
    "Priority Scheduling (Non-Preemptive)",
    "Priority Scheduling (Preemptive)",
    "Round Robin", 
    "Multi-Level Queue Scheduling"
])

num = st.number_input("Number of Processes", min_value=1, max_value=10, value=3)
process_list = []
for i in range(num):
    cols = st.columns(4)
    with cols[0]: pid = st.number_input(f"Process ID {i+1}", value=i+1, key=f"pid{i}")
    with cols[1]: arrival = st.number_input(f"Arrival Time {i+1}", min_value=0, value=0, key=f"arr{i}")
    with cols[2]: burst = st.number_input(f"Burst Time {i+1}", min_value=1, value=1, key=f"bur{i}")
    priority = 0
    queue = 1
    if "Priority Scheduling" in selected:
        with cols[3]: priority = st.number_input(f"Priority {i+1} (lower = higher priority)", min_value=0, value=0, key=f"pri{i}")
    if selected == "Multi-Level Queue Scheduling":
        queue = st.number_input(f"Queue for Process {i+1}", min_value=1, value=1, key=f"que{i}")
    process_list.append({'process_id': pid, 'arrival': arrival, 'burst': burst, 'priority': priority, 'queue': queue})

definitions = {}
if selected == "Multi-Level Queue Scheduling":
    levels = sorted({p['queue'] for p in process_list})
    for lvl in levels:
        alg = st.selectbox(f"Algorithm for Queue {lvl}", ["First Come First Serve", "Round Robin", "Priority Scheduling (Non-Preemptive)"], key=f"qal{lvl}")
        if alg == "First Come First Serve": definitions[lvl] = {'algorithm': 'FCFS'}
        elif alg == "Round Robin":
            q = st.number_input(f"Time Quantum for Queue {lvl}", min_value=1, value=2, key=f"qq{lvl}")
            definitions[lvl] = {'algorithm': 'RR', 'quantum': q}
        else: definitions[lvl] = {'algorithm': 'PR-NP'}

quantum = 0
if selected == "Round Robin": quantum = st.number_input("Time Quantum", min_value=1, value=2)

if st.button("Simulate"):
    if selected == "First Come First Serve": timeline = first_come_first_served(process_list)
    elif selected == "Shortest Job First (Non-Preemptive)": timeline = shortest_job_first_non_preemptive(process_list)
    elif selected == "Shortest Job First (Preemptive)": timeline = shortest_job_first_preemptive(process_list)
    elif selected == "Priority Scheduling (Non-Preemptive)": timeline = priority_non_preemptive(process_list)
    elif selected == "Priority Scheduling (Preemptive)": timeline = priority_preemptive(process_list)
    elif selected == "Round Robin": timeline = round_robin_scheduling(process_list, quantum)
    else: timeline = multi_level_queue_scheduling(process_list, definitions)

    avg_metrics, _ = compute_performance_metrics(timeline, process_list)
    st.metric("Average Turnaround Time", f"{avg_metrics['turnaround']:.2f}")
    st.metric("Average Waiting Time", f"{avg_metrics['waiting']:.2f}")
    st.metric("Average Response Time", f"{avg_metrics['response']:.2f}")

    st.subheader("Schedule Details")
    st.dataframe(pd.DataFrame(timeline))

    fig, ax = plt.subplots(figsize=(15, 4))
    for seg in timeline:
        ax.add_patch(Rectangle((seg['start_time'], 0), seg['finish_time'] - seg['start_time'], 1,
                               color='skyblue', edgecolor='black'))
        ax.text((seg['start_time'] + seg['finish_time']) / 2, 0.5,
                f"P{seg['process_id']}", ha='center', va='center', fontsize=10)
    ax.set_xlim(0, max(seg['finish_time'] for seg in timeline) + 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time", fontsize=12)
    ax.set_title("Gantt Chart of Process Execution", fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    st.pyplot(fig)
