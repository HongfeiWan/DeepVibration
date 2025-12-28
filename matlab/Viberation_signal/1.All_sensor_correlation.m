%% 多振动传感器数据相关性分析
% 类似于Joint_HPGe_Viber_Energy_Amplitude.m，可以选出指定时间段内的振动传感器mat文件
% 然后将这些所有日期的mat文件数据一起进行相关性分析
% 支持多个传感器（detector1, detector2, detector3, detector4, detector5）

%% 1. 设置时间范围和传感器选择
% 设置分析的时间范围
starttime = datetime('2025-05-15 00:00:00.000', 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');
endtime = datetime('2025-06-10 00:00:00.000', 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');

% 选择要分析的传感器编号（1-5）
selected_sensors = [3];  % 可以修改为只选择特定的传感器

fprintf('分析时间范围: %s 到 %s\n', datestr(starttime), datestr(endtime));
fprintf('选择的传感器: %s\n', mat2str(selected_sensors));

%% 2. 获取时间范围内的所有日期
% 生成时间范围内的所有日期
all_dates = starttime:days(1):endtime;
unique_dates = unique(dateshift(all_dates, 'start', 'day'));
fprintf('需要分析的日期数量: %d\n', length(unique_dates));

%% 3. 读取所有选定传感器的数据
% 初始化存储所有数据的结构
all_sensor_data = struct();
success_count = 0;

% 遍历每个传感器
for sensor_idx = 1:length(selected_sensors)
    sensor_num = selected_sensors(sensor_idx);
    fprintf('\n正在处理传感器 %d...\n', sensor_num);
    
    % 初始化该传感器的数据存储
    all_sensor_data.(sprintf('detector%d', sensor_num)) = struct();
    all_sensor_data.(sprintf('detector%d', sensor_num)).time = [];
    all_sensor_data.(sprintf('detector%d', sensor_num)).temperature = [];
    all_sensor_data.(sprintf('detector%d', sensor_num)).velocity_x = [];
    all_sensor_data.(sprintf('detector%d', sensor_num)).velocity_y = [];
    all_sensor_data.(sprintf('detector%d', sensor_num)).velocity_z = [];
    all_sensor_data.(sprintf('detector%d', sensor_num)).acceleration_x = [];
    all_sensor_data.(sprintf('detector%d', sensor_num)).acceleration_y = [];
    all_sensor_data.(sprintf('detector%d', sensor_num)).acceleration_z = [];
    all_sensor_data.(sprintf('detector%d', sensor_num)).displacement_x = [];
    all_sensor_data.(sprintf('detector%d', sensor_num)).displacement_y = [];
    all_sensor_data.(sprintf('detector%d', sensor_num)).displacement_z = [];
    all_sensor_data.(sprintf('detector%d', sensor_num)).frequency_x = [];
    all_sensor_data.(sprintf('detector%d', sensor_num)).frequency_y = [];
    all_sensor_data.(sprintf('detector%d', sensor_num)).frequency_z = [];
    
    % 遍历每个日期，读取对应的detector文件
    for date_idx = 1:length(unique_dates)
        current_date = unique_dates(date_idx);
        date_str = datestr(current_date, 'yyyymmdd');
        filename = sprintf('detector%d%s.mat', sensor_num, date_str);
        filepath = fullfile('../data', filename);
        
        % 检查文件是否存在
        if exist(filepath, 'file')
            fprintf('  正在读取文件: %s\n', filename);
            try
                % 加载.mat文件
                loaded_data = load(filepath);
                var_names = fieldnames(loaded_data);
                
                % 获取数据变量
                if length(var_names) == 1
                    detector_data = loaded_data.(var_names{1});
                else
                    detector_var_idx = find(contains(var_names, 'detector', 'IgnoreCase', true) | ...
                                           contains(var_names, 'data', 'IgnoreCase', true));
                    if ~isempty(detector_var_idx)
                        detector_data = loaded_data.(var_names{detector_var_idx(1)});
                    else
                        detector_data = loaded_data.(var_names{1});
                    end
                end
                
                % 处理数据并添加到总数据中
                fprintf('  数据类型: %s\n', class(detector_data));
                if istable(detector_data)
                    % 从第2行开始提取数据（跳过标题行）
                    if height(detector_data) > 1
                        data_start_row = 2;
                        data_end_row = height(detector_data);
                        w = width(detector_data);
                        fprintf('  表格列数: %d\n', w);
                        if w < 14
                            fprintf('  列数不足14，跳过该文件\n');
                            continue;
                        end
                        
                        % 直接提取为数组，避免后续再做表索引
                        time_arr = detector_data{data_start_row:data_end_row, 1};
                        temp_arr = detector_data{data_start_row:data_end_row, 2};
                        vel_x_arr = detector_data{data_start_row:data_end_row, 3};
                        vel_y_arr = detector_data{data_start_row:data_end_row, 4};
                        vel_z_arr = detector_data{data_start_row:data_end_row, 5};
                        acc_x_arr = detector_data{data_start_row:data_end_row, 6};
                        acc_y_arr = detector_data{data_start_row:data_end_row, 7};
                        acc_z_arr = detector_data{data_start_row:data_end_row, 8};
                        disp_x_arr = detector_data{data_start_row:data_end_row, 9};
                        disp_y_arr = detector_data{data_start_row:data_end_row, 10};
                        disp_z_arr = detector_data{data_start_row:data_end_row, 11};
                        freq_x_arr = detector_data{data_start_row:data_end_row, 12};
                        freq_y_arr = detector_data{data_start_row:data_end_row, 13};
                        freq_z_arr = detector_data{data_start_row:data_end_row, 14};
                        
                        % 转换时间格式并筛选
                        time_array = time_arr;
                        if isdatetime(time_array)
                            time_dt = time_array;
                        elseif isnumeric(time_array)
                            epoch0 = datetime('1970-01-01', 'TimeZone', '');
                            time_dt = epoch0 + seconds(time_array);
                        else
                            try
                                time_strings_fixed = strrep(time_array, ':', '.');
                                time_dt = datetime(time_strings_fixed, 'InputFormat', 'yyyy-MM-dd HH.mm.ss.SSS');
                            catch
                                fprintf('    时间转换失败，跳过此文件\n');
                                continue;
                            end
                        end
                        
                        % 筛选时间范围内的数据
                        time_idx = time_dt >= starttime & time_dt <= endtime;
                        if any(time_idx)
                            % 将筛选后的数据添加到总数据中
                            all_sensor_data.(sprintf('detector%d', sensor_num)).time = ...
                                [all_sensor_data.(sprintf('detector%d', sensor_num)).time; time_dt(time_idx)];
                            all_sensor_data.(sprintf('detector%d', sensor_num)).temperature = ...
                                [all_sensor_data.(sprintf('detector%d', sensor_num)).temperature; get_numeric_vector(temp_arr(time_idx))];
                            all_sensor_data.(sprintf('detector%d', sensor_num)).velocity_x = ...
                                [all_sensor_data.(sprintf('detector%d', sensor_num)).velocity_x; get_numeric_vector(vel_x_arr(time_idx))];
                            all_sensor_data.(sprintf('detector%d', sensor_num)).velocity_y = ...
                                [all_sensor_data.(sprintf('detector%d', sensor_num)).velocity_y; get_numeric_vector(vel_y_arr(time_idx))];
                            all_sensor_data.(sprintf('detector%d', sensor_num)).velocity_z = ...
                                [all_sensor_data.(sprintf('detector%d', sensor_num)).velocity_z; get_numeric_vector(vel_z_arr(time_idx))];
                            all_sensor_data.(sprintf('detector%d', sensor_num)).acceleration_x = ...
                                [all_sensor_data.(sprintf('detector%d', sensor_num)).acceleration_x; get_numeric_vector(acc_x_arr(time_idx))];
                            all_sensor_data.(sprintf('detector%d', sensor_num)).acceleration_y = ...
                                [all_sensor_data.(sprintf('detector%d', sensor_num)).acceleration_y; get_numeric_vector(acc_y_arr(time_idx))];
                            all_sensor_data.(sprintf('detector%d', sensor_num)).acceleration_z = ...
                                [all_sensor_data.(sprintf('detector%d', sensor_num)).acceleration_z; get_numeric_vector(acc_z_arr(time_idx))];
                            all_sensor_data.(sprintf('detector%d', sensor_num)).displacement_x = ...
                                [all_sensor_data.(sprintf('detector%d', sensor_num)).displacement_x; get_numeric_vector(disp_x_arr(time_idx))];
                            all_sensor_data.(sprintf('detector%d', sensor_num)).displacement_y = ...
                                [all_sensor_data.(sprintf('detector%d', sensor_num)).displacement_y; get_numeric_vector(disp_y_arr(time_idx))];
                            all_sensor_data.(sprintf('detector%d', sensor_num)).displacement_z = ...
                                [all_sensor_data.(sprintf('detector%d', sensor_num)).displacement_z; get_numeric_vector(disp_z_arr(time_idx))];
                            all_sensor_data.(sprintf('detector%d', sensor_num)).frequency_x = ...
                                [all_sensor_data.(sprintf('detector%d', sensor_num)).frequency_x; get_numeric_vector(freq_x_arr(time_idx))];
                            all_sensor_data.(sprintf('detector%d', sensor_num)).frequency_y = ...
                                [all_sensor_data.(sprintf('detector%d', sensor_num)).frequency_y; get_numeric_vector(freq_y_arr(time_idx))];
                            all_sensor_data.(sprintf('detector%d', sensor_num)).frequency_z = ...
                                [all_sensor_data.(sprintf('detector%d', sensor_num)).frequency_z; get_numeric_vector(freq_z_arr(time_idx))];
                            
                            fprintf('    成功添加 %d 个数据点\n', sum(time_idx));
                        end
                    end
                end
                
                success_count = success_count + 1;
                
            catch ME
                fprintf('    读取文件 %s 时出错: %s\n', filename, ME.message);
            end
        else
            fprintf('  文件不存在: %s\n', filename);
        end
    end
end

fprintf('\n总共成功读取了 %d 个文件\n', success_count);

%% 4. 按传感器分别进行14维相关性分析（仿照 Vibration_Correlation.m）
fprintf('\n开始按传感器分别进行14维相关性分析...\n');

column_names_sensor = {'时间', '温度', '速度X', '速度Y', '速度Z', ...
                       '加速度X', '加速度Y', '加速度Z', '位移X', '位移Y', '位移Z', ...
                       '频率X', '频率Y', '频率Z'};

for sensor_idx = 1:length(selected_sensors)
    sensor_num = selected_sensors(sensor_idx);
    sensor_name = sprintf('detector%d', sensor_num);
    
    if ~isfield(all_sensor_data, sensor_name)
        fprintf('传感器 %d 无数据结构，跳过。\n', sensor_num);
        continue;
    end
    
    S = all_sensor_data.(sensor_name);
    % 检查长度
    len_time = length(S.time);
    len_vecs = [length(S.temperature), length(S.velocity_x), length(S.velocity_y), length(S.velocity_z), ...
                length(S.acceleration_x), length(S.acceleration_y), length(S.acceleration_z), ...
                length(S.displacement_x), length(S.displacement_y), length(S.displacement_z), ...
                length(S.frequency_x), length(S.frequency_y), length(S.frequency_z)];
    
    min_len = min([len_time, len_vecs]);
    if isempty(min_len) || min_len == 0
        fprintf('传感器 %d 有空数据，跳过。\n', sensor_num);
        continue;
    end
    
    if any(len_vecs ~= len_time)
        % 对齐到最短长度，避免长度不一致
        fprintf('传感器 %d 各列长度不一致，按最短长度对齐：%d\n', sensor_num, min_len);
    end
    
    % 构建14列数据矩阵
    data_matrix_sensor = zeros(min_len, 14);
    % 时间列
    try
        data_matrix_sensor(:, 1) = datenum(S.time(1:min_len));
    catch
        % 万一时间不是datetime，尝试转换
        data_matrix_sensor(:, 1) = get_numeric_vector(S.time(1:min_len));
    end
    % 其他列
    data_matrix_sensor(:, 2)  = double(S.temperature(1:min_len));
    data_matrix_sensor(:, 3)  = double(S.velocity_x(1:min_len));
    data_matrix_sensor(:, 4)  = double(S.velocity_y(1:min_len));
    data_matrix_sensor(:, 5)  = double(S.velocity_z(1:min_len));
    data_matrix_sensor(:, 6)  = double(S.acceleration_x(1:min_len));
    data_matrix_sensor(:, 7)  = double(S.acceleration_y(1:min_len));
    data_matrix_sensor(:, 8)  = double(S.acceleration_z(1:min_len));
    data_matrix_sensor(:, 9)  = double(S.displacement_x(1:min_len));
    data_matrix_sensor(:, 10) = double(S.displacement_y(1:min_len));
    data_matrix_sensor(:, 11) = double(S.displacement_z(1:min_len));
    data_matrix_sensor(:, 12) = double(S.frequency_x(1:min_len));
    data_matrix_sensor(:, 13) = double(S.frequency_y(1:min_len));
    data_matrix_sensor(:, 14) = double(S.frequency_z(1:min_len));
    
    % 计算相关性矩阵
    corr_mat_sensor = corrcoef(data_matrix_sensor, 'Rows', 'complete');
    
    % 绘图
    figure('Position', [100, 100, 1200, 1000]);
    imagesc(corr_mat_sensor);
    colorbar; colormap('jet');
    xticks(1:14); yticks(1:14);
    xticklabels(column_names_sensor); yticklabels(column_names_sensor);
    xtickangle(45);
    title(sprintf('传感器 %d 振动数据相关性矩阵', sensor_num), 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('变量', 'FontSize', 12); ylabel('变量', 'FontSize', 12);
    
    % 标注强相关
    for i = 1:14
        for j = 1:14
            if abs(corr_mat_sensor(i, j)) > 0
                text(j, i, sprintf('%.2f', corr_mat_sensor(i, j)), ...
                    'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', 'white');
            end
        end
    end
    
    % 保存
    try
        filename = sprintf('../images/sensor%d_correlation_%s_to_%s.png', sensor_num, ...
            datestr(starttime, 'yyyy-mm-dd'), datestr(endtime, 'yyyy-mm-dd'));
        saveas(gcf, filename);
        fprintf('传感器 %d 相关性热图已保存: %s\n', sensor_num, filename);
    catch ME
        fprintf('保存传感器 %d 热图时出错: %s\n', sensor_num, ME.message);
    end
    
    % 文本输出强相关对
    fprintf('\n传感器 %d 强相关性变量对 (|r| > 0.7):\n', sensor_num);
    for i = 1:14
        for j = i+1:14
            if abs(corr_mat_sensor(i, j)) > 0.7
                fprintf('%s 与 %s: r = %.3f\n', column_names_sensor{i}, column_names_sensor{j}, corr_mat_sensor(i, j));
            end
        end
    end
end

% 单独分析完成后直接结束脚本，避免执行联合合并分析
return;

%% 本地函数：将table子集稳健转换为列向量（double）
function vec = get_numeric_vector(tbl_slice)
% 支持以下情况：
% - table 切片（单列或多列 -> 取第一列）
% - cellstr / string 数组（尝试str2double）
% - datetime（转为datenum）
% - 已是double
% - 其他类型尽量转换，失败则返回空

    vec = [];
    try
        if istable(tbl_slice)
            if width(tbl_slice) > 1
                tbl_slice = tbl_slice(:,1);
            end
            arr = tbl_slice{:,1};
        else
            arr = tbl_slice;
        end

        if isdatetime(arr)
            vec = datenum(arr);
        elseif isduration(arr)
            vec = seconds(arr);
        elseif isnumeric(arr)
            vec = double(arr(:));
        elseif iscellstr(arr) || isstring(arr) || iscell(arr)
            try
                if iscell(arr)
                    arr = string(arr);
                end
                vec = str2double(arr(:));
            catch
                vec = [];
            end
        else
            try
                vec = double(arr(:));
            catch
                vec = [];
            end
        end
    catch
        vec = [];
    end
end
