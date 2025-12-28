function Vibration_Amplitude_Temperature()
    % 振动位移与温度关系分析脚本
    % 绘制双轴图：左轴显示温度，右轴显示振动位移
    % 参考 Vibration_show_all.m 的数据获取方法
    
    %% 1. 设置参数
    % 选择要分析的传感器编号（1-5）
    selected_sensor = 5;  % 可以修改为选择特定的传感器
    
    % 设置分析的时间范围
    startDate = datetime(2025, 5, 28);
    endDate = datetime(2025, 6, 03);
    
    % 颜色设置
    temp_color = 'red';      % 温度曲线颜色
    disp_color = 'blue';     % 位移曲线颜色
    
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
    sensor_data.displacement_x = [];
    sensor_data.displacement_y = [];
    sensor_data.displacement_z = [];
    
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
                            % 提取温度和位移数据
                            time_arr = detector_data{data_start_row:data_end_row, 1};
                            temp_arr = detector_data{data_start_row:data_end_row, 2};
                            disp_x_arr = detector_data{data_start_row:data_end_row, 9};
                            disp_y_arr = detector_data{data_start_row:data_end_row, 10};
                            disp_z_arr = detector_data{data_start_row:data_end_row, 11};
                            
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
                                sensor_data.displacement_x = [sensor_data.displacement_x; get_numeric_vector(disp_x_arr(time_idx))];
                                sensor_data.displacement_y = [sensor_data.displacement_y; get_numeric_vector(disp_y_arr(time_idx))];
                                sensor_data.displacement_z = [sensor_data.displacement_z; get_numeric_vector(disp_z_arr(time_idx))];
                                
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
    data_lengths = [length(sensor_data.temperature), length(sensor_data.displacement_x), ...
                    length(sensor_data.displacement_y), length(sensor_data.displacement_z)];
    
    min_len = min([length(sensor_data.time), data_lengths]);
    fprintf('数据点数量: %d\n', min_len);
    
    % 性能参数：每条曲线最大绘制点数（超出将抽稀）
    max_points_per_series = 2000000;
    
    %% 5. 创建双轴图：温度（左轴）和振动位移（右轴）
    figure('Name', sprintf('传感器%d - 温度与振动位移关系', selected_sensor), ...
           'Position', [100, 100, 1400, 600]);
    set(gcf, 'Renderer', 'opengl', 'GraphicsSmoothing', 'off');
    
    % 创建左轴（温度）
    yyaxis left
    [time_temp_ds, temp_ds] = thin_series(sensor_data.time(1:min_len), sensor_data.temperature(1:min_len), max_points_per_series);
    temp_plot = plot(time_temp_ds, temp_ds, ...
                     'Color', temp_color, 'LineWidth', 2, 'DisplayName', '温度');
    ylabel('温度 (℃)', 'FontSize', 14, 'FontWeight', 'bold');
    ylim_temp = [min(sensor_data.temperature(1:min_len)) - 2, ...
                 max(sensor_data.temperature(1:min_len)) + 2];
    ylim(ylim_temp);
    % 使左轴颜色与温度曲线颜色一致
    ax = gca;
    ax.YAxis(1).Color = temp_color;
    
    % 创建右轴（位移）
    yyaxis right
    hold on;
    [time_dx, disp_x_ds] = thin_series(sensor_data.time(1:min_len), sensor_data.displacement_x(1:min_len), max_points_per_series);
    [time_dy, disp_y_ds] = thin_series(sensor_data.time(1:min_len), sensor_data.displacement_y(1:min_len), max_points_per_series);
    [time_dz, disp_z_ds] = thin_series(sensor_data.time(1:min_len), sensor_data.displacement_z(1:min_len), max_points_per_series);
    disp_x_plot = plot(time_dx, disp_x_ds, '.', 'Color', disp_color, 'MarkerSize', 12, 'DisplayName', '位移X');
    disp_y_plot = plot(time_dy, disp_y_ds, '.', 'Color', [0.5, 0, 0.5], 'MarkerSize', 12, 'DisplayName', '位移Y');
    disp_z_plot = plot(time_dz, disp_z_ds, '.', 'Color', [0, 0.5, 0.5], 'MarkerSize', 12, 'DisplayName', '位移Z');
    hold off;
    ylabel('振动位移 (μm)', 'FontSize', 14, 'FontWeight', 'bold');
    
    % 设置右轴颜色
    ax = gca;
    ax.YAxis(2).Color = disp_color;
    
    % 设置标题和标签
    title(sprintf('传感器%d - 温度与振动位移关系图 (%s 至 %s)', selected_sensor, ...
          datestr(startDate, 'yyyy-mm-dd'), datestr(endDate, 'yyyy-mm-dd')), ...
          'FontSize', 16, 'FontWeight', 'bold');
    xlabel('时间', 'FontSize', 14, 'FontWeight', 'bold');
    
    % 添加图例
    legend([temp_plot, disp_x_plot, disp_y_plot, disp_z_plot], ...
           'Location', 'best', 'FontSize', 12);
    
    % 设置网格
    grid on;
    ax.FontSize = 12;
    
    % 直接使用 datetime 轴（更快），并设置显示范围
    xlim([sensor_data.time(1), sensor_data.time(min_len)]);
    
    %% 6. 保存图像
    try
        % 保存主图
        figure(1);
        filename_main = sprintf('../images/sensor%d_temp_disp_main_%s_to_%s.png', selected_sensor, ...
            datestr(startDate, 'yyyy-mm-dd'), datestr(endDate, 'yyyy-mm-dd'));
        saveas(gcf, filename_main);
        fprintf('主图已保存: %s\n', filename_main);
        
    catch ME
        fprintf('保存图像时出错: %s\n', ME.message);
    end
    
    fprintf('\n分析完成！已创建温度与振动位移的双轴时间序列图\n');
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

%% 本地函数：时间序列抽稀
function [x_ds, y_ds] = thin_series(x, y, max_points)
% 简单均匀抽样，将序列长度限制在 max_points 以内
    x = x(:); y = y(:);
    n = numel(x);
    if n <= max_points || max_points <= 0
        x_ds = x; y_ds = y; return;
    end
    idx = round(linspace(1, n, max_points));
    x_ds = x(idx);
    y_ds = y(idx);
end
