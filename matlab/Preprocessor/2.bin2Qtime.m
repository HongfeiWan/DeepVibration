% 定义文件路径和文件名
read_path = '/st0/share/data/raw_data/CEvNS_DZL_test_sanmen/20250520/';
save_path = '/st0/home/wanhf/Time_Energy';
filename_input = '20250520_CEvNS_DZL_sm_pre10000_tri10mV_SA6us0.8x50_SA12us0.8x50_TAout10us1.2x100_TAout10us0.5x3_RT50mHz_NaISA1us1.0x20_plasticsci1-10_bkg'; % 默认文件名，可以根据实际情况修改

RUN_Start_NUMBER = 281; % 起始运行编号
RUN_End_NUMBER = 281; % 结束运行编号

% 定义通道数和事件数
CHANNEL_NUMBER = 4; % 只处理前4个通道
EVENT_NUMBER = 10000; % 每个bin文件中的事件数
MAX_WINDOWS = 30000; % 时间窗 120μs

% 保存选项配置
SAVE_OPTIONS = struct();
SAVE_OPTIONS.use_compression = false;       % 是否使用压缩（设为false以加速保存）
SAVE_OPTIONS.save_in_batches = false;       % 是否分批保存
SAVE_OPTIONS.batch_size = 1000;             % 每批事件数
SAVE_OPTIONS.save_only_processed = true;    % 只保存处理后的数据（时间数据和Q值）
SAVE_OPTIONS.use_fast_format = false;       % 是否优先使用快速格式（设为false，因为大数据文件需要v7.3）

% 尝试启动并行计算池
try
    if isempty(gcp('nocreate'))
        parpool('local');
    end
    use_parallel = true;
    fprintf('使用并行计算处理数据\n');
catch ME
    use_parallel = false;
    fprintf('无法启动并行计算池，将使用串行处理: %s\n', ME.message);
end

% 主程序：并行处理多个文件
if use_parallel
    parfor i = RUN_Start_NUMBER:RUN_End_NUMBER
        % 构造文件名
        run_filename = sprintf('%s%sFADC_RAW_Data_%d.bin', read_path, filename_input, i);
        try
            process_single_file_optimized(run_filename, CHANNEL_NUMBER, EVENT_NUMBER, SAVE_OPTIONS, save_path);
        catch ME
            fprintf('处理文件 %s 时出错: %s\n', run_filename, ME.message);
        end
    end
else
    for i = RUN_Start_NUMBER:RUN_End_NUMBER
        % 构造文件名
        run_filename = sprintf('%s%sFADC_RAW_Data_%d.bin', read_path, filename_input, i);
        
        try
            process_single_file_optimized(run_filename, CHANNEL_NUMBER, EVENT_NUMBER, SAVE_OPTIONS, save_path);
        catch ME
            fprintf('处理文件 %s 时出错: %s\n', run_filename, ME.message);
        end
    end
end

fprintf('所有文件处理完成！\n');

% 定义优化的处理单个文件的函数
function process_single_file_optimized(run_filename, CHANNEL_NUMBER, EVENT_NUMBER, SAVE_OPTIONS, save_path)
    fprintf('============================================\n');
    fprintf('Opening %s\n', run_filename);
    
    % 打开文件
    fid = fopen(run_filename, 'r');
    if fid == -1
        error('Failed to open file: %s', run_filename);
    end
    
    % 读取 Run Header
    pstt = fread(fid, 1, 'double'); % Program Start Time
    fprintf('Program Start Time: %f s.\n', pstt);
    
    % 读取 V1725-1 Channel DAC (只读取前4个通道)
    V1725_1_DAC = fread(fid, 16, 'uint32'); % 仍然读取16个通道的DAC值
    fprintf('V1725-1 Channel DAC: ');
    fprintf('%d ', V1725_1_DAC);
    fprintf('\n');
    
    % 读取其他 Run Header 信息
    V1725_1_twd = fread(fid, 1, 'uint32'); % Time Window
    fprintf('V1725-1 Time Window: %d\n', V1725_1_twd);
    V1725_1_pretg = fread(fid, 1, 'uint32'); % Pre Trigger
    fprintf('V1725-1 Pre Trigger: %d\n', V1725_1_pretg);
    V1725_1_opch = fread(fid, 1, 'uint32'); % Opened Channel
    fprintf('V1725-1 Opened Channel: %d\n', V1725_1_opch);
    
    % 读取 Run Start Time
    rstt = fread(fid, 1, 'double');
    fprintf('Run Start Time: %f s.\n', rstt);
    
    % 初始化事件数据结构 - 使用单独的数组
    idevt = zeros(1, EVENT_NUMBER);
    trig = zeros(1, EVENT_NUMBER);
    time = zeros(1, EVENT_NUMBER);
    deadtime = zeros(1, EVENT_NUMBER);
    % 启用Q值计算
    ped = zeros(CHANNEL_NUMBER, EVENT_NUMBER);
    pedt = zeros(CHANNEL_NUMBER, EVENT_NUMBER);
    q = zeros(CHANNEL_NUMBER, EVENT_NUMBER);
    max_val = zeros(CHANNEL_NUMBER, EVENT_NUMBER);
    maxpt = zeros(CHANNEL_NUMBER, EVENT_NUMBER);
    min_val = zeros(CHANNEL_NUMBER, EVENT_NUMBER);
    minpt = zeros(CHANNEL_NUMBER, EVENT_NUMBER);
    tb = zeros(CHANNEL_NUMBER, EVENT_NUMBER);
    rms = zeros(CHANNEL_NUMBER, EVENT_NUMBER);
    Qp = zeros(CHANNEL_NUMBER, EVENT_NUMBER);
    
    % 预分配事件数据数组
    hit_pat_array = zeros(EVENT_NUMBER, 1);
    v1729_tg_rec_array = zeros(EVENT_NUMBER, 1);
    evt_endtime_array = zeros(EVENT_NUMBER, 1);
    v1725_1_tgno_array = zeros(EVENT_NUMBER, 1);
    v1725_1_tag_array = zeros(EVENT_NUMBER, 1);
    
    % 根据保存选项决定数据结构
    if SAVE_OPTIONS.save_only_processed
        % 只保存处理后的数据，不保存原始脉冲
        channel_data = [];
        time_data = zeros(EVENT_NUMBER, 1);
    else
        % 保存原始脉冲数据
        channel_data = zeros(V1725_1_twd, CHANNEL_NUMBER, EVENT_NUMBER, 'uint16');
        time_data = zeros(EVENT_NUMBER, 1);
    end
    
    % 首先读取所有事件的数据
    actual_events = 0;
    for j = 1:EVENT_NUMBER
        % 检查是否到达文件末尾
        if feof(fid)
            fprintf('警告：文件在事件 %d 处结束，实际读取了 %d 个事件\n', j, actual_events);
            break;
        end
        
        % 读取 Event Header
        hit_pat_data = fread(fid, 1, 'uint32'); % Hit_pat
        if isempty(hit_pat_data)
            fprintf('警告：文件在事件 %d 处结束，实际读取了 %d 个事件\n', j, actual_events);
            break;
        end
        hit_pat_array(j) = hit_pat_data;
        
        v1729_tg_rec_data = fread(fid, 1, 'uint32'); % V1729_tg_rec
        if isempty(v1729_tg_rec_data)
            fprintf('警告：文件在事件 %d 处结束，实际读取了 %d 个事件\n', j, actual_events);
            break;
        end
        v1729_tg_rec_array(j) = v1729_tg_rec_data;
        
        evt_endtime_data = fread(fid, 1, 'uint32'); % Evt_endtime
        if isempty(evt_endtime_data)
            fprintf('警告：文件在事件 %d 处结束，实际读取了 %d 个事件\n', j, actual_events);
            break;
        end
        evt_endtime_array(j) = evt_endtime_data;
        
        v1725_1_tgno_data = fread(fid, 1, 'uint32'); % V1725_1_tgno
        if isempty(v1725_1_tgno_data)
            fprintf('警告：文件在事件 %d 处结束，实际读取了 %d 个事件\n', j, actual_events);
            break;
        end
        v1725_1_tgno_array(j) = v1725_1_tgno_data;
        
        v1725_1_tag_data = fread(fid, 1, 'uint32'); % V1725_1_tag
        if isempty(v1725_1_tag_data)
            fprintf('警告：文件在事件 %d 处结束，实际读取了 %d 个事件\n', j, actual_events);
            break;
        end
        v1725_1_tag_array(j) = v1725_1_tag_data;
        
        % 读取每个通道的数据 (只处理前4个通道，但需要跳过其他通道)
        for k = 1:16 % 仍然需要读取所有16个通道的数据
            channel_data_temp = fread(fid, V1725_1_twd, 'uint16');
            if isempty(channel_data_temp)
                fprintf('警告：文件在事件 %d 通道 %d 处结束，实际读取了 %d 个事件\n', j, k, actual_events);
                break;
            end
            % 只保存前4个通道的数据
            if k <= CHANNEL_NUMBER && ~SAVE_OPTIONS.save_only_processed
                channel_data(:, k, j) = channel_data_temp;
            end
            % 即使只保存处理后的数据，也需要临时保存前4个通道的数据用于Q值计算
            if k <= CHANNEL_NUMBER && SAVE_OPTIONS.save_only_processed
                if ~exist('temp_channel_data', 'var')
                    temp_channel_data = zeros(V1725_1_twd, CHANNEL_NUMBER, EVENT_NUMBER, 'uint16');
                end
                temp_channel_data(:, k, j) = channel_data_temp;
            end
        end
        
        % 计算时间信息（将在后续处理中加上pstt）
        time_with_pstt = (evt_endtime_data / 1000); % 先转换为秒，后续会加上pstt
        time_data(j) = time_with_pstt;
        
        actual_events = actual_events + 1;
    end
    
    % 更新实际事件数并调整数组大小
    if actual_events < EVENT_NUMBER
        EVENT_NUMBER = actual_events;
        fprintf('实际读取的事件数：%d\n', EVENT_NUMBER);
        
        % 调整数组大小以匹配实际事件数
        hit_pat_array = hit_pat_array(1:actual_events);
        v1729_tg_rec_array = v1729_tg_rec_array(1:actual_events);
        evt_endtime_array = evt_endtime_array(1:actual_events);
        v1725_1_tgno_array = v1725_1_tgno_array(1:actual_events);
        v1725_1_tag_array = v1725_1_tag_array(1:actual_events);
        if ~SAVE_OPTIONS.save_only_processed
            channel_data = channel_data(:, :, 1:actual_events);
        end
        time_data = time_data(1:actual_events);
        
        % 重新初始化数组以匹配实际事件数
        idevt = idevt(1:EVENT_NUMBER);
        trig = trig(1:EVENT_NUMBER);
        time = time(1:EVENT_NUMBER);
        deadtime = deadtime(1:EVENT_NUMBER);
        % 启用Q值计算数组调整
        ped = ped(:, 1:EVENT_NUMBER);
        pedt = pedt(:, 1:EVENT_NUMBER);
        q = q(:, 1:EVENT_NUMBER);
        max_val = max_val(:, 1:EVENT_NUMBER);
        maxpt = maxpt(:, 1:EVENT_NUMBER);
        min_val = min_val(:, 1:EVENT_NUMBER);
        minpt = minpt(:, 1:EVENT_NUMBER);
        tb = tb(:, 1:EVENT_NUMBER);
        rms = rms(:, 1:EVENT_NUMBER);
        Qp = Qp(:, 1:EVENT_NUMBER);
    end
    % 关闭文件
    fclose(fid);
    
    % 处理所有事件（串行处理，避免parfor内部嵌套问题）
    for j = 1:EVENT_NUMBER
        % 获取预读取的事件头信息
        Hit_pat = hit_pat_array(j);
        V1729_tg_rec = v1729_tg_rec_array(j);
        Evt_endtime = evt_endtime_array(j);
        V1725_1_tgno = v1725_1_tgno_array(j);
        V1725_1_tag = v1725_1_tag_array(j);
        % 计算时间
        TTTV1725 = bitand(uint32(V1725_1_tag), uint32(0x7FFFFFFF));
        TimeV1725 = 10 * double(TTTV1725);
        % 存储事件信息
        idevt(j) = EVENT_NUMBER * 0 + j; % 简化ID计算
        trig(j) = V1725_1_tgno;
        time(j) = Evt_endtime / 1000; % 转换为秒
        deadtime(j) = 0; % 假设 deadtime 为 0，根据实际情况修改
        % 处理每个通道的数据
        for k = 1:CHANNEL_NUMBER
            if SAVE_OPTIONS.save_only_processed
                % 使用临时保存的脉冲数据进行Q值计算
                V1725_1_pulse = temp_channel_data(:, k, j);
            else
                V1725_1_pulse = channel_data(:, k, j);
            end
            
            if size(V1725_1_pulse, 1) == 1
                V1725_1_pulse = V1725_1_pulse';
            end
            
            % 启用Q值计算
            fit_start = 6000;
            FIT_RANGE = 5500;
            PED_RANGE_FP = 1000;

            pulse_length = length(V1725_1_pulse);
            if (k ~= 2 && k ~= 3)
                ped(k, j) = sum(double(V1725_1_pulse(1:min(1000, pulse_length)))) / 1000;
                pedt(k, j) = sum(double(V1725_1_pulse(max(1, pulse_length-999):pulse_length))) / 1000;
            else
                ped(k, j) = sum(double(V1725_1_pulse(1:min(PED_RANGE_FP, pulse_length)))) / PED_RANGE_FP;
                pedt_range_start = fit_start + FIT_RANGE - PED_RANGE_FP;
                pedt_range_end = fit_start + FIT_RANGE - 1;
                if pedt_range_start <= pulse_length && pedt_range_end <= pulse_length && pedt_range_start <= pedt_range_end
                    pedt(k, j) = sum(double(V1725_1_pulse(pedt_range_start:pedt_range_end))) / PED_RANGE_FP;
                else
                    pedt(k, j) = 0;
                end
            end
            time_weights = ((1:pulse_length) - 4000)';

            tb(k, j) = sum(time_weights .* double(V1725_1_pulse));
            q(k, j) = sum(double(V1725_1_pulse));
            [max_val(k, j), maxpt(k, j)] = max(double(V1725_1_pulse));
            min_val(k, j) = 65535;
            minpt(k, j) = 0;
            for l = 1:pulse_length
                if double(V1725_1_pulse(l)) < min_val(k, j)
                    if (k > 5 && (l < V1725_1_pretg - 3500 || l > V1725_1_pretg - 1000))
                        continue;
                    else
                        min_val(k, j) = double(V1725_1_pulse(l));
                        minpt(k, j) = l;
                    end
                end
            end
            rms(k, j) = sqrt(sum(double(V1725_1_pulse) .^ 2) / pulse_length - (q(k, j) * q(k, j)) / (pulse_length * pulse_length));

            Qp_RANDN = [900, 1800, 500, 3000]; % 只保留前4个通道的参数
            Qp_RANUP = [1500, 3000, 4500, 5500]; % 只保留前4个通道的参数

            Qp(k, j) = 0;
            if k >= 2 && Qp_RANUP(k) > 0
                qp_start = max(1, Qp_RANDN(k));
                qp_end = min(pulse_length, Qp_RANUP(k));
                if qp_start <= qp_end
                    Qp(k, j) = sum(double(V1725_1_pulse(qp_start:qp_end)));
                end
            elseif k < 2
                qp_start = max(1, maxpt(k, j) - Qp_RANDN(k));
                qp_end = min(pulse_length, maxpt(k, j) + Qp_RANUP(k));
                if qp_start <= qp_end
                    Qp(k, j) = sum(double(V1725_1_pulse(qp_start:qp_end)));
                end
            end
        end
    end % 结束事件处理循环
    
    % 清理临时变量以节省内存
    if SAVE_OPTIONS.save_only_processed && exist('temp_channel_data', 'var')
        clear temp_channel_data;
    end
    
    % 更新时间数据（加上pstt）
    time_data = time_data + pstt;
    
    % 保存文件 - 只保存时间数据和Q值
    [~, filename, ~] = fileparts(run_filename);
    base_filename = fullfile(save_path, sprintf('%s_Time_Q', filename));

    % 确保保存目录存在
    if ~exist(save_path, 'dir')
        mkdir(save_path);
        fprintf('创建保存目录: %s\n', save_path);
    end

    fprintf('开始保存时间数据和Q值...\n');

    tic;

    % 只保存时间数据和Q值
    save(sprintf('%s.mat', base_filename), 'time_data', 'q', '-v7.3');
    fprintf('时间数据和Q值保存完成，耗时: %.2f秒\n', toc);

    % 显示文件大小信息
    if exist(sprintf('%s.mat', base_filename), 'file')
        fileInfo = dir(sprintf('%s.mat', base_filename));
        fileSizeMB = fileInfo.bytes / (1024^2);
        fprintf('文件大小: %.2f MB\n', fileSizeMB);
    end

    fprintf('数据保存完成！\n');
end 