function Vibration_show_all()
    % 振动传感器信息展示脚本
    % 展示指定detector的所有信息，分为多个figure窗口
    % 参考 Vibration_sensor_All_Information_in_One_Plot.m 和 Vibration_all_sensor_correlation.m
    
    %% 1. 设置参数
    % 选择要分析的传感器编号（1-5）
    selected_sensor = 2;  % 可以修改为选择特定的传感器
    
    % 设置分析的时间范围
    startDate = datetime(2025, 5, 28);
    endDate = datetime(2025, 6, 03);
    
    % 颜色和线宽设置
    colors = ['r','g','b','c','m','y','k'];     % 红绿蓝青洋黄黑
    lineWidth = 1.5;     % 线宽
    
    fprintf('选择的传感器: %d\n', selected_sensor);
    fprintf('分析时间范围: %s 到 %s\n', datestr(startDate), datestr(endDate));
    
    %% 2. 获取时间范围内的所有日期
    all_dates = startDate:days(1):endDate;
    unique_dates = unique(dateshift(all_dates, 'start', 'day'));
    fprintf('需要分析的日期数量: %d\n', length(unique_dates));
    
    %% 3. 读取指定传感器的数据
    fprintf('\n正在读取传感器 %d 的数据...\n', selected_sensor);
    
    % 初始化数据存储
    sensor_data = struct();
    sensor_data.time = [];
    sensor_data.temperature = [];
    sensor_data.velocity_x = [];
    sensor_data.velocity_y = [];
    sensor_data.velocity_z = [];
    sensor_data.acceleration_x = [];
    sensor_data.acceleration_y = [];
    sensor_data.acceleration_z = [];
    sensor_data.displacement_x = [];
    sensor_data.displacement_y = [];
    sensor_data.displacement_z = [];
    sensor_data.frequency_x = [];
    sensor_data.frequency_y = [];
    sensor_data.frequency_z = [];
    
    success_count = 0;
    
    % 遍历每个日期，读取对应的detector文件
    for date_idx = 1:length(unique_dates)
        current_date = unique_dates(date_idx);
        date_str = datestr(current_date, 'yyyymmdd');
        filename = sprintf('detector%d%s.mat', selected_sensor, date_str);
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
                
                % 处理数据
                if istable(detector_data)
                    % 从第2行开始提取数据（跳过标题行）
                    if height(detector_data) > 1
                        data_start_row = 2;
                        data_end_row = height(detector_data);
                        w = width(detector_data);
                        
                        if w >= 14
                            % 提取各列数据
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
                            
                            % 转换时间格式
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
                            time_idx = time_dt >= startDate & time_dt <= endDate;
                            if any(time_idx)
                                % 将筛选后的数据添加到总数据中
                                sensor_data.time = [sensor_data.time; time_dt(time_idx)];
                                sensor_data.temperature = [sensor_data.temperature; get_numeric_vector(temp_arr(time_idx))];
                                sensor_data.velocity_x = [sensor_data.velocity_x; get_numeric_vector(vel_x_arr(time_idx))];
                                sensor_data.velocity_y = [sensor_data.velocity_y; get_numeric_vector(vel_y_arr(time_idx))];
                                sensor_data.velocity_z = [sensor_data.velocity_z; get_numeric_vector(vel_z_arr(time_idx))];
                                sensor_data.acceleration_x = [sensor_data.acceleration_x; get_numeric_vector(acc_x_arr(time_idx))];
                                sensor_data.acceleration_y = [sensor_data.acceleration_y; get_numeric_vector(acc_y_arr(time_idx))];
                                sensor_data.acceleration_z = [sensor_data.acceleration_z; get_numeric_vector(acc_z_arr(time_idx))];
                                sensor_data.displacement_x = [sensor_data.displacement_x; get_numeric_vector(disp_x_arr(time_idx))];
                                sensor_data.displacement_y = [sensor_data.displacement_y; get_numeric_vector(disp_y_arr(time_idx))];
                                sensor_data.displacement_z = [sensor_data.displacement_z; get_numeric_vector(disp_z_arr(time_idx))];
                                sensor_data.frequency_x = [sensor_data.frequency_x; get_numeric_vector(freq_x_arr(time_idx))];
                                sensor_data.frequency_y = [sensor_data.frequency_y; get_numeric_vector(freq_y_arr(time_idx))];
                                sensor_data.frequency_z = [sensor_data.frequency_z; get_numeric_vector(freq_z_arr(time_idx))];
                                
                                fprintf('    成功添加 %d 个数据点\n', sum(time_idx));
                                success_count = success_count + 1;
                            end
                        else
                            fprintf('    列数不足14，跳过该文件\n');
                        end
                    end
                end
                
            catch ME
                fprintf('    读取文件 %s 时出错: %s\n', filename, ME.message);
            end
        else
            fprintf('  文件不存在: %s\n', filename);
        end
    end
    
    fprintf('\n总共成功读取了 %d 个文件\n', success_count);
    
    %% 4. 检查数据完整性
    if isempty(sensor_data.time)
        fprintf('错误：没有读取到有效数据！\n');
        return;
    end
    
    % 检查各列数据长度
    data_lengths = [length(sensor_data.temperature), length(sensor_data.velocity_x), ...
                    length(sensor_data.velocity_y), length(sensor_data.velocity_z), ...
                    length(sensor_data.acceleration_x), length(sensor_data.acceleration_y), ...
                    length(sensor_data.acceleration_z), length(sensor_data.displacement_x), ...
                    length(sensor_data.displacement_y), length(sensor_data.displacement_z), ...
                    length(sensor_data.frequency_x), length(sensor_data.frequency_y), ...
                    length(sensor_data.frequency_z)];
    
    min_len = min([length(sensor_data.time), data_lengths]);
    fprintf('数据点数量: %d\n', min_len);
    
    %% 5. 创建多个figure展示不同类型的数据
    
    % Figure 1: 温度数据
    figure('Name', sprintf('传感器%d - 温度数据', selected_sensor), ...
           'Position', [100, 100, 1200, 400]);
    plot(sensor_data.time(1:min_len), sensor_data.temperature(1:min_len), ...
         'Color', colors(1), 'LineWidth', lineWidth);
    title(sprintf('传感器%d - 温度变化趋势', selected_sensor), 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('时间', 'FontSize', 12);
    ylabel('温度 (℃)', 'FontSize', 12);
    grid on;
    ax = gca;
    ax.FontSize = 12;
    
    % Figure 2: 速度数据 (X, Y, Z)
    figure('Name', sprintf('传感器%d - 速度数据', selected_sensor), ...
           'Position', [200, 200, 1200, 400]);
    hold on;
    plot(sensor_data.time(1:min_len), sensor_data.velocity_x(1:min_len), ...
         'Color', colors(1), 'LineWidth', lineWidth, 'DisplayName', '速度X');
    plot(sensor_data.time(1:min_len), sensor_data.velocity_y(1:min_len), ...
         'Color', colors(2), 'LineWidth', lineWidth, 'DisplayName', '速度Y');
    plot(sensor_data.time(1:min_len), sensor_data.velocity_z(1:min_len), ...
         'Color', colors(3), 'LineWidth', lineWidth, 'DisplayName', '速度Z');
    hold off;
    title(sprintf('传感器%d - 速度数据 (X, Y, Z)', selected_sensor), 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('时间', 'FontSize', 12);
    ylabel('速度 (mm/s)', 'FontSize', 12);
    legend('Location', 'best');
    grid on;
    ax = gca;
    ax.FontSize = 12;
    
    % Figure 3: 加速度数据 (X, Y, Z)
    figure('Name', sprintf('传感器%d - 加速度数据', selected_sensor), ...
           'Position', [300, 300, 1200, 400]);
    hold on;
    plot(sensor_data.time(1:min_len), sensor_data.acceleration_x(1:min_len), ...
         'Color', colors(1), 'LineWidth', lineWidth, 'DisplayName', '加速度X');
    plot(sensor_data.time(1:min_len), sensor_data.acceleration_y(1:min_len), ...
         'Color', colors(2), 'LineWidth', lineWidth, 'DisplayName', '加速度Y');
    plot(sensor_data.time(1:min_len), sensor_data.acceleration_z(1:min_len), ...
         'Color', colors(3), 'LineWidth', lineWidth, 'DisplayName', '加速度Z');
    hold off;
    title(sprintf('传感器%d - 加速度数据 (X, Y, Z)', selected_sensor), 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('时间', 'FontSize', 12);
    ylabel('加速度 (g*m/s^2)', 'FontSize', 12);
    legend('Location', 'best');
    grid on;
    ax = gca;
    ax.FontSize = 12;
    
    % Figure 4: 位移数据 (X, Y, Z)
    figure('Name', sprintf('传感器%d - 位移数据', selected_sensor), ...
           'Position', [400, 400, 1200, 400]);
    hold on;
    plot(sensor_data.time(1:min_len), sensor_data.displacement_x(1:min_len), ...
         'Color', colors(1), 'LineWidth', lineWidth, 'DisplayName', '位移X');
    plot(sensor_data.time(1:min_len), sensor_data.displacement_y(1:min_len), ...
         'Color', colors(2), 'LineWidth', lineWidth, 'DisplayName', '位移Y');
    plot(sensor_data.time(1:min_len), sensor_data.displacement_z(1:min_len), ...
         'Color', colors(3), 'LineWidth', lineWidth, 'DisplayName', '位移Z');
    hold off;
    title(sprintf('传感器%d - 位移数据 (X, Y, Z)', selected_sensor), 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('时间', 'FontSize', 12);
    ylabel('位移 (μm)', 'FontSize', 12);
    legend('Location', 'best');
    grid on;
    ax = gca;
    ax.FontSize = 12;
    
    % Figure 5: 频率数据 (X, Y, Z)
    figure('Name', sprintf('传感器%d - 频率数据', selected_sensor), ...
           'Position', [500, 500, 1200, 400]);
    hold on;
    plot(sensor_data.time(1:min_len), sensor_data.frequency_x(1:min_len), ...
         'Color', colors(1), 'LineWidth', lineWidth, 'DisplayName', '频率X');
    plot(sensor_data.time(1:min_len), sensor_data.frequency_y(1:min_len), ...
         'Color', colors(2), 'LineWidth', lineWidth, 'DisplayName', '频率Y');
    plot(sensor_data.time(1:min_len), sensor_data.frequency_z(1:min_len), ...
         'Color', colors(3), 'LineWidth', lineWidth, 'DisplayName', '频率Z');
    hold off;
    title(sprintf('传感器%d - 频率数据 (X, Y, Z)', selected_sensor), 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('时间', 'FontSize', 12);
    ylabel('频率 (Hz)', 'FontSize', 12);
    legend('Location', 'best');
    grid on;
    ax = gca;
    ax.FontSize = 12;
    
    % Figure 6: 数据统计信息
    figure('Name', sprintf('传感器%d - 数据统计', selected_sensor), ...
           'Position', [600, 600, 800, 600]);
    
    % 创建统计表格
    data_names = {'温度(℃)', '速度X(mm/s)', '速度Y(mm/s)', '速度Z(mm/s)', ...
                  '加速度X(g*m/s²)', '加速度Y(g*m/s²)', '加速度Z(g*m/s²)', ...
                  '位移X(μm)', '位移Y(μm)', '位移Z(μm)', ...
                  '频率X(Hz)', '频率Y(Hz)', '频率Z(Hz)'};
    
    data_values = [sensor_data.temperature(1:min_len), sensor_data.velocity_x(1:min_len), ...
                   sensor_data.velocity_y(1:min_len), sensor_data.velocity_z(1:min_len), ...
                   sensor_data.acceleration_x(1:min_len), sensor_data.acceleration_y(1:min_len), ...
                   sensor_data.acceleration_z(1:min_len), sensor_data.displacement_x(1:min_len), ...
                   sensor_data.displacement_y(1:min_len), sensor_data.displacement_z(1:min_len), ...
                   sensor_data.frequency_x(1:min_len), sensor_data.frequency_y(1:min_len), ...
                   sensor_data.frequency_z(1:min_len)];
    
    stats_table = zeros(4, 13);
    for i = 1:13
        valid_data = data_values(:, i);
        valid_data = valid_data(~isnan(valid_data) & ~isinf(valid_data));
        if ~isempty(valid_data)
            stats_table(1, i) = mean(valid_data);      % 均值
            stats_table(2, i) = std(valid_data);       % 标准差
            stats_table(3, i) = min(valid_data);       % 最小值
            stats_table(4, i) = max(valid_data);       % 最大值
        end
    end
    
    % 显示统计表格
    uitable('Data', stats_table, ...
            'ColumnName', data_names, ...
            'RowName', {'均值', '标准差', '最小值', '最大值'}, ...
            'Units', 'normalized', ...
            'Position', [0.1, 0.1, 0.8, 0.8]);
    
    title(sprintf('传感器%d - 数据统计信息', selected_sensor), 'FontSize', 16, 'FontWeight', 'bold');
    
    fprintf('\n已创建 %d 个figure窗口展示传感器 %d 的所有信息\n', 6, selected_sensor);
    fprintf('包括：温度、速度、加速度、位移、频率数据和统计信息\n');
    
    %% 6. 保存图像
    try
        for fig_num = 1:6
            figure(fig_num);
            filename = sprintf('../images/sensor%d_figure%d_%s_to_%s.png', selected_sensor, fig_num, ...
                datestr(startDate, 'yyyy-mm-dd'), datestr(endDate, 'yyyy-mm-dd'));
            saveas(gcf, filename);
            fprintf('Figure %d 已保存: %s\n', fig_num, filename);
        end
    catch ME
        fprintf('保存图像时出错: %s\n', ME.message);
    end
end

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
