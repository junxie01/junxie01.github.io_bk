import os
import sys
import shutil
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees
from obspy.core.util import AttribDict

# --- 配置参数 ---
USGS_CLIENT = Client("USGS")
IRIS_CLIENT = Client("IRIS")
MIN_MAGNITUDE = 6.0
CHECK_WINDOW_HOURS = 12   # 检索过去12小时的地震
TARGET_DURATION = 2 * 3600  # 目标下载时长（2小时）
OUTPUT_ROOT = "/Users/junxie/Work/auto_download/earthquake_data"

# 选取主要的全球台网 (IU, II, CU, IC, GE)
NETWORKS = "IU,II,CU,IC,GE"

if not os.path.exists(OUTPUT_ROOT):
    os.makedirs(OUTPUT_ROOT)

def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def get_download_info(event_dir):
    """读取已存在的下载状态文件"""
    status_file = os.path.join(event_dir, "download_status.txt")
    if not os.path.exists(status_file):
        return None
    try:
        info = {}
        with open(status_file, "r") as f:
            for line in f:
                if ":" in line:
                    key, val = line.split(":", 1)
                    info[key.strip()] = val.strip()
        return info
    except:
        return None

def write_download_info(event_dir, station_count, duration):
    """写入下载状态文件"""
    status_file = os.path.join(event_dir, "download_status.txt")
    with open(status_file, "w") as f:
        f.write(f"Station Count: {station_count}\n")
        f.write(f"Duration Seconds: {duration}\n")
        f.write(f"Download Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def process_event(event):
    """处理单个地震事件：下载、转换SAC、写入头变量、绘图"""
    origin = event.origins[0]
    otime = origin.time
    evla = origin.latitude
    evlo = origin.longitude
    evdp = origin.depth / 1000.0 if origin.depth else 0.0  # km
    mag = event.magnitudes[0].mag
    
    # 文件夹命名格式: YYYYMMDDHHMMSSM6.1
    folder_name = f"{otime.strftime('%Y%m%d%H%M%S')}M{mag:.1f}"
    event_dir = os.path.join(OUTPUT_ROOT, folder_name)
    
    now = UTCDateTime.now()
    # 如果地震刚刚发生，计算从发震到现在的时间
    available_duration = min(TARGET_DURATION, now - otime)
    
    if os.path.exists(event_dir):
        info = get_download_info(event_dir)
        if info:
            prev_duration = float(info.get("Duration Seconds", 0))
            if prev_duration >= TARGET_DURATION:
                log(f"跳过已处理的完整地震数据: {folder_name}")
                return
            elif available_duration <= prev_duration:
                log(f"当前可用时长 ({available_duration:.0f}s) 未显著增加，跳过: {folder_name}")
                return
            else:
                log(f"数据不完整 ({prev_duration:.0f}s < {TARGET_DURATION}s)，且有新数据，重新下载...")
                shutil.rmtree(event_dir)
        else:
            log(f"目录已存在但无状态文件，重新处理: {folder_name}")
            shutil.rmtree(event_dir)
    
    os.makedirs(event_dir)
    log(f"正在处理地震: {folder_name} (时长: {available_duration:.0f}s)")

    try:
        # 1. 下载波形和响应
        log("从 IRIS 下载数据...")
        st = IRIS_CLIENT.get_waveforms(
            network=NETWORKS, station="*", location="*", channel="BH?",
            starttime=otime, endtime=otime + available_duration,
            attach_response=True
        )
        
        inventory = IRIS_CLIENT.get_stations(
            network=NETWORKS, station="*", channel="BH?",
            starttime=otime, endtime=otime + available_duration,
            level="channel"
        )
    except Exception as e:
        log(f"数据下载失败: {e}")
        # 清理不完整的文件夹
        if os.path.exists(event_dir):
            shutil.rmtree(event_dir)
        return

    # 2. 预处理
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.05)
    
    pre_filt = [0.005, 0.01, 5.0, 10.0]
    log("正在去仪器响应 (转为速度)...")
    st.remove_response(output="VEL", pre_filt=pre_filt)

    # 3. 写入 SAC 头变量并保存文件
    processed_traces = []
    sac_dir = os.path.join(event_dir, "SAC_files")
    os.makedirs(sac_dir)

    for tr in st:
        try:
            net = tr.stats.network
            sta = tr.stats.station
            cha = tr.stats.channel
            coords = inventory.get_coordinates(tr.id, datetime=otime)
            dist_deg = locations2degrees(evla, evlo, coords['latitude'], coords['longitude'])
            tr.stats.distance = dist_deg
            
            if not hasattr(tr.stats, 'sac'):
                tr.stats.sac = AttribDict()
            
            tr.stats.sac.evla = evla
            tr.stats.sac.evlo = evlo
            tr.stats.sac.evdp = evdp
            tr.stats.sac.mag = mag
            tr.stats.sac.o = 0.0
            tr.stats.sac.stla = coords['latitude']
            tr.stats.sac.stlo = coords['longitude']
            tr.stats.sac.stel = coords['elevation']
            tr.stats.sac.garc = dist_deg
            
            sac_filename = f"{net}_{sta}_{cha}.SAC"
            tr.write(os.path.join(sac_dir, sac_filename), format="SAC")
            processed_traces.append(tr)
        except:
            continue

    if not processed_traces:
        log("没有可用的台站记录。")
        shutil.rmtree(event_dir)
        return

    # 写入状态说明文件
    write_download_info(event_dir, len(processed_traces), available_duration)

    # 4. 可视化
    st_processed = Stream(traces=processed_traces)
    bands = [{"name": "0.01-0.1Hz", "fmin": 0.01, "fmax": 0.1},
             {"name": "0.1-2Hz", "fmin": 0.1, "fmax": 2.0}]
    for band in bands:
        plot_path = os.path.join(event_dir, f"plot_{band['name'].replace('-', '_')}.png")
        plot_record_section(st_processed, band, plot_path, folder_name)

def plot_record_section(st, band, save_path, title):
    st_plot = st.copy()
    st_plot.filter("bandpass", freqmin=band['fmin'], freqmax=band['fmax'])
    fig, axes = plt.subplots(1, 3, figsize=(18, 12), sharey=True)
    comp_chars = ['Z', 'N', 'E']
    for i, char in enumerate(comp_chars):
        ax = axes[i]
        comp_st = Stream()
        for tr in st_plot:
            c = tr.stats.component.upper()
            if c.endswith(char) or (char == 'N' and c.endswith('1')) or (char == 'E' and c.endswith('2')):
                comp_st.append(tr)
        traces = sorted(comp_st, key=lambda x: x.stats.distance)
        for tr in traces:
            data = tr.data
            if np.max(np.abs(data)) > 0:
                data = (data / np.max(np.abs(data))) * 0.8
            ax.plot(tr.times(), data + tr.stats.distance, 'k', lw=0.4)
        ax.set_title(f"{char} Component [{band['name']}]")
        ax.set_xlabel("Time (s)")
        if i == 0: ax.set_ylabel("Distance (degree)")
    plt.suptitle(f"Event: {title}", fontsize=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close()
    log(f"已生成图表: {save_path}")

def main():
    log("开始执行定期检索...")
    endtime = UTCDateTime.now()
    starttime = endtime - (CHECK_WINDOW_HOURS * 3600)
    try:
        catalog = USGS_CLIENT.get_events(starttime=starttime, endtime=endtime, minmagnitude=MIN_MAGNITUDE)
        log(f"在过去 {CHECK_WINDOW_HOURS} 小时内发现 {len(catalog)} 个 > M{MIN_MAGNITUDE} 的地震。")
        for event in catalog:
            process_event(event)
    except Exception as e:
        log(f"检索失败: {e}")

if __name__ == "__main__":
    main()
