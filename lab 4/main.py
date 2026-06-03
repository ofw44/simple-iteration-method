import numpy as np
import matplotlib.pyplot as plt

def simulate_autoscaling(rps_normal, rps_spike, rps_per_server,
                         cpu_threshold, scale_up_count,
                         delay, total_time, dt=1):
    
    n_steps = int(total_time / dt) + 1
    time = np.linspace(0, total_time, n_steps)
    
    servers = np.zeros(n_steps)
    rps = np.zeros(n_steps)
    
    servers[0] = 3
    pending_queue = []
    
    for i in range(n_steps):
        t = time[i]
        
        if t < 10:
            rps[i] = rps_normal
        elif t < 20:
            rps[i] = rps_normal + (rps_spike - rps_normal) * (t - 10) / 10
        else:
            rps[i] = rps_spike
        
        if i == 0:
            continue
        
        servers[i] = servers[i-1]
        
        still_pending = []
        for ready_time, count in pending_queue:
            if t >= ready_time:
                servers[i] += count
            else:
                still_pending.append((ready_time, count))
        pending_queue = still_pending
        
        current_capacity = servers[i] * rps_per_server
        cpu = (rps[i] / current_capacity * 100) if current_capacity > 0 else 999
        
        if cpu > cpu_threshold:
            pending_queue.append((t + delay, scale_up_count))
        
        elif cpu < cpu_threshold * 0.3 and servers[i] > 3:
            can_remove = 0
            for j in range(i-1, max(0, i-300), -1):
                if servers[j] < servers[j+1]:
                    break
                can_remove += 1
            if can_remove > 60:
                servers[i] = max(3, servers[i] - 1)
    
    return time, servers, rps

scenarios = [
    {'name': 'Задержка 30с', 'delay': 30, 'color': 'blue', 'linewidth': 3, 'linestyle': '-'},
    {'name': 'Задержка 60с', 'delay': 60, 'color': 'green', 'linewidth': 3, 'linestyle': '--'},
    {'name': 'Задержка 120с', 'delay': 120, 'color': 'red', 'linewidth': 3, 'linestyle': '-.'},
]

fig, ax = plt.subplots(figsize=(16, 8))

for scenario in scenarios:
    time, servers, rps = simulate_autoscaling(
        rps_normal=300,
        rps_spike=1200,
        rps_per_server=100,
        cpu_threshold=70,
        scale_up_count=2,
        delay=scenario['delay'],
        total_time=800,
        dt=1
    )
    
    ax.plot(time, servers, color=scenario['color'], linewidth=scenario['linewidth'],
            linestyle=scenario['linestyle'], label=scenario['name'])

ax.axvline(x=10, color='black', linestyle=':', alpha=0.5, linewidth=1)
ax.axvline(x=20, color='black', linestyle=':', alpha=0.5, linewidth=1)
ax.text(15, ax.get_ylim()[1]*0.95, 'Рост\nтрафика', ha='center', fontsize=11, fontweight='bold')

ax.set_xlabel('Время (секунды)', fontsize=13)
ax.set_ylabel('Количество серверов', fontsize=13)
ax.set_title('Автоскейлинг Kubernetes: влияние задержки запуска подов на стабильность', fontsize=15)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, loc='upper left')
ax.set_xlim([0, 800])

plt.tight_layout()
plt.show()

olli_scenarios = [
    {'name': 'Консервативный (порог 80%, +1 сервер, задержка 60с)', 
     'threshold': 80, 'scale_up': 1, 'delay': 60,
     'color': 'green', 'linewidth': 3, 'linestyle': '-'},
    {'name': 'Агрессивный (порог 60%, +3 сервера, задержка 90с)', 
     'threshold': 60, 'scale_up': 3, 'delay': 90,
     'color': 'orange', 'linewidth': 3, 'linestyle': '--'},
    {'name': 'Опасный (порог 50%, +5 серверов, задержка 120с)', 
     'threshold': 50, 'scale_up': 5, 'delay': 120,
     'color': 'red', 'linewidth': 3, 'linestyle': '-.'}
]

fig, ax = plt.subplots(figsize=(16, 8))

for scenario in olli_scenarios:
    time, servers, rps = simulate_autoscaling(
        rps_normal=300,
        rps_spike=1200,
        rps_per_server=100,
        cpu_threshold=scenario['threshold'],
        scale_up_count=scenario['scale_up'],
        delay=scenario['delay'],
        total_time=1000,
        dt=1
    )
    
    ax.plot(time, servers, color=scenario['color'], linewidth=scenario['linewidth'],
            linestyle=scenario['linestyle'], label=scenario['name'])

ax.axvline(x=10, color='black', linestyle=':', alpha=0.5, linewidth=1)
ax.axvline(x=20, color='black', linestyle=':', alpha=0.5, linewidth=1)

ax.set_xlabel('Время (секунды)', fontsize=13)
ax.set_ylabel('Количество серверов', fontsize=13)
ax.set_title('Эффект Олли: перелеты и раскачка при агрессивном автоскейлинге', fontsize=15)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, loc='upper left')
ax.set_xlim([0, 1000])

plt.tight_layout()
plt.show()