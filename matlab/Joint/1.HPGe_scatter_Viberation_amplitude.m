%% 设置导入选项
opts = delimitedTextImportOptions("NumVariables", 2);
% 指定范围和分隔符
opts.DataLines = [1, Inf];
opts.Delimiter = ",";
% 指定列名称和类型
opts.VariableNames = ["Time", "Value"];
opts.VariableTypes = ["double", "double"];
% 指定文件级属性
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

%% 获取目录下所有txt文件
folderPath = "../data/Detector_txt";
txtFiles = dir(fullfile(folderPath, '*.txt'));

%% 循环读取每个txt文件并导入数据
for i = 1:length(txtFiles)
    fileName = txtFiles(i).name; % 获取文件名
    filePath = fullfile(folderPath, fileName); % 获取文件完整路径
    % 以文件名（去掉扩展名）命名变量
    varName = strrep(fileName, '.txt', ''); % 去掉文件扩展名
    % 动态分配变量名并导入数据
    assignin('base', varName, readtable(filePath, opts));
end

%% 清除临时变量
clear opts folderPath txtFiles i fileName filePath varName

%% 对t_max处理
t_max_0510.Time=t_max_0510.Time+3829705251.828405-2.082816000000000e+09;
t_max_0516.Time=t_max_0516.Time+3830232437.925693-2.082816000000000e+09;
t_max_0520.Time=t_max_0520.Time+3830569388.496263-2.082816000000000e+09;
t_max_0510.Value=t_max_0510.Value*0.00084199-0.83341;
t_max_0516.Value=t_max_0516.Value*0.00084199-0.83341;
t_max_0520.Value=t_max_0520.Value*0.00084199-0.83341;
if ~exist('t_max_0510', 'var') || ~exist('t_max_0520', 'var') || ~exist('t_max_0516', 'var')
    error('变量不存在，请确保它们已正确加载到工作区。');
end
% 确保两个表的结构一致
if ~isequal(t_max_0510.Properties.VariableNames, t_max_0516.Properties.VariableNames,t_max_0520.Properties.VariableNames)
    error('变量名不一致，无法合并。');
end
% 合并表
HPGe_signal = vertcat(t_max_0510, t_max_0516 , t_max_0520);
% 清除原始变量（可选）
clear t_max_0510 t_max_0520 t_max_0516

%% 时间转换
epochStart = datetime('1970-01-01', 'TimeZone', '');
data_transformed = epochStart + seconds(HPGe_signal.Time);
data_transformed.Format = 'yyyy-MM-dd HH:mm:ss.SSS';
clear epochStart

%% 筛选能区、时间
%filteredIdx = HPGe_signal.Value > 0 & HPGe_signal.Value < 0.5;%单独筛选能量
starttime = datetime('2025-05-30 00:00:00.000', 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');
endtime = datetime('2025-06-03 00:00:00.000', 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');
energyrange=[0,1];%0-0.5keV
filteredIdx = (HPGe_signal.Value > energyrange(1) & HPGe_signal.Value < energyrange(2)) & ...
    (data_transformed >= starttime & data_transformed <= endtime);

%% 根据筛选时间读取对应的detector22025xxxx.mat文件
% 获取筛选时间范围内的所有日期
filtered_dates = data_transformed(filteredIdx);
unique_dates = unique(dateshift(filtered_dates, 'start', 'day'));

% 逐个读取对应日期的detector2文件，并直接存入工作区
success_count = 0;

% 遍历每个日期，读取对应的detector2文件
for i = 1:length(unique_dates)
    current_date = unique_dates(i);
    date_str = datestr(current_date, 'yyyymmdd');
    filename = sprintf('detector2%s.mat', date_str);
    filepath = fullfile('../data', filename);
    % 检查文件是否存在
    if exist(filepath, 'file')
        fprintf('正在读取文件: %s\n', filename);
        try
            % 加载.mat文件
            loaded_data = load(filepath);
            % 获取加载的变量名
            var_names = fieldnames(loaded_data);
            % 假设只有一个变量，或者我们可以指定变量名
            if length(var_names) == 1
                detector_data = loaded_data.(var_names{1});
            else
                % 如果有多个变量，可以选择特定的变量名
                % 这里假设变量名包含'detector'或'data'
                detector_var_idx = find(contains(var_names, 'detector', 'IgnoreCase', true) | ...
                                       contains(var_names, 'data', 'IgnoreCase', true));
                if ~isempty(detector_var_idx)
                    detector_data = loaded_data.(var_names{detector_var_idx(1)});
                else
                    detector_data = loaded_data.(var_names{1});
                end
            end
            % 将数据直接存入工作区变量（使用文件名去掉扩展名）
            filebase = erase(filename, '.mat');
            assignin('base', filebase, detector_data);
            success_count = success_count + 1;
            % 显示当前文件的数据信息
            if isnumeric(detector_data)
                fprintf('  - 数据大小: %s\n', mat2str(size(detector_data)));
            elseif isstruct(detector_data)
                fprintf('  - 结构体字段: %s\n', strjoin(fieldnames(detector_data), ', '));
            else
                fprintf('  - 数据类型: %s\n', class(detector_data));
            end
        catch ME
            fprintf('读取文件 %s 时出错: %s\n', filename, ME.message);
        end
    else
        fprintf('文件不存在: %s\n', filename);
    end
end
% 显示读取结果
fprintf('成功读取了 %d 个日期的detector2数据\n', success_count);
clear loaded_data var_names detector_data detector_var_idx filebase success_count current_date date_str filepath;

% 画图（双轴叠加：左轴HPGe，右轴detector2）
figure; % 创建新图形窗口

% 先收集所有detector2数据
all_det_time = [];
all_det_value = [];
fprintf('开始收集detector2数据，共有 %d 个日期\n', length(unique_dates));

for i = 1:length(unique_dates)
    current_date = unique_dates(i);
    date_str = datestr(current_date, 'yyyymmdd');
    var_name = ['detector2' date_str];
    
    fprintf('检查变量: %s\n', var_name);
    
    if evalin('base', sprintf('exist(''%s'',''var'')', var_name))
        fprintf('  变量存在，正在读取...\n');
        det_data = evalin('base', var_name);
        
        % 显示数据类型和大小
        fprintf('  数据类型: %s, 大小: %s\n', class(det_data), mat2str(size(det_data)));
        
        % 尝试解析时间与数值
        det_time = [];
        det_value = [];
        
        if istable(det_data)
            vns = det_data.Properties.VariableNames;
            fprintf('  表格变量名: %s\n', strjoin(vns, ', '));
            
            % 检查第一列是否为时间格式
            first_col = det_data.(vns{1});
            if ischar(first_col) || isstring(first_col) || isdatetime(first_col)
                % 第一列是时间，第二列是数值
                det_time = det_data.(vns{1});
                det_value = det_data.(vns{2});
                fprintf('  从表格中提取到时间和数值数据（第1、2列）\n');
            else
                fprintf('  表格中第1列不是时间格式\n');
            end
        elseif isstruct(det_data)
            fns = fieldnames(det_data);
            fprintf('  结构体字段: %s\n', strjoin(fns, ', '));
            if any(strcmpi(fns,'Time')) && any(strcmpi(fns,'Value'))
                det_time = det_data.(fns{find(strcmpi(fns,'Time'),1)});
                det_value = det_data.(fns{find(strcmpi(fns,'Value'),1)});
                fprintf('  从结构体中提取到时间和数值数据\n');
            else
                fprintf('  结构体中未找到Time或Value字段\n');
            end
        elseif isnumeric(det_data) && size(det_data,2) >= 2
            det_time = det_data(:,1);
            det_value = det_data(:,2);
            fprintf('  从数值矩阵中提取到时间和数值数据\n');
        else
            fprintf('  无法识别的数据格式\n');
        end
        
        % 将时间转换为datetime并筛选
        if ~isempty(det_time) && ~isempty(det_value)
            fprintf('  时间数据长度: %d, 数值数据长度: %d\n', length(det_time), length(det_value));
            
            if isnumeric(det_time)
                epoch0 = datetime('1970-01-01', 'TimeZone', '');
                det_time_dt = epoch0 + seconds(det_time);
                det_time_dt.Format = 'yyyy-MM-dd HH:mm:ss.SSS';
                fprintf('  时间转换完成，范围: %s 到 %s\n', string(det_time_dt(1)), string(det_time_dt(end)));
            elseif isdatetime(det_time)
                det_time_dt = det_time;
                fprintf('  时间已为datetime格式\n');
            elseif ischar(det_time) || isstring(det_time)
                % 字符串时间格式，尝试转换为datetime
                try
                    det_time_dt = datetime(det_time, 'InputFormat', 'yyyy-MM-dd HH:mm:ss:SSS');
                    fprintf('  字符串时间转换完成，范围: %s 到 %s\n', string(det_time_dt(1)), string(det_time_dt(end)));
                catch
                    fprintf('  字符串时间格式转换失败\n');
                    det_time_dt = [];
                end
            else
                det_time_dt = [];
                fprintf('  时间格式无法处理\n');
            end
            
            if ~isempty(det_time_dt)
                idx_t = det_time_dt >= starttime & det_time_dt <= endtime;
                fprintf('  筛选时间范围 %s 到 %s，符合条件的点数: %d\n', string(starttime), string(endtime), sum(idx_t));
                if any(idx_t)
                    all_det_time = [all_det_time; det_time_dt(idx_t)];
                    all_det_value = [all_det_value; det_value(idx_t)];
                    fprintf('  已添加到总数据中\n');
                end
            end
        else
            fprintf('  时间或数值数据为空\n');
        end
    else
        fprintf('  变量不存在\n');
    end
end

fprintf('收集完成，总共收集到 %d 个数据点\n', length(all_det_time));

% 绘制左轴HPGe数据
yyaxis left;
h1 = scatter(data_transformed(filteredIdx), HPGe_signal.Value(filteredIdx), 'red', 'filled', SizeData=2);
ax = gca;
ax.YColor = 'red'; % 设置左轴颜色为红色

% 绘制右轴detector2数据
yyaxis right;
h2 = [];
if ~isempty(all_det_time)
    h2 = scatter(all_det_time, all_det_value, 'blue', 'filled', SizeData=1);
end
ax.YColor = 'blue'; % 设置右轴颜色为蓝色

hold on;

% 设置坐标轴标签字体和字体大小
ax = gca; % 获取当前坐标轴对象
FontSize = 25;

xticklabels(datestr(ax.XAxis.TickValues, 'YYYY-mm-dd')); % 设置 X 轴刻度标签为 MM-dd 格式

ax.XLabel.FontName = 'Times New Roman'; % 设置 X 轴标签字体
ax.XAxis.FontName = 'Times New Roman'; % 设置 X 轴刻度标签字体
for k = 1:numel(ax.YAxis)
    ax.YAxis(k).FontName = 'Times New Roman'; % 设置两侧 Y 轴刻度标签字体
end

ax.XLabel.FontSize = FontSize; % 设置 X 轴标签字体大小
ax.XAxis.FontSize = FontSize-5; % 设置 X 轴刻度标签字体大小
for k = 1:numel(ax.YAxis)
    ax.YAxis(k).FontSize = FontSize-5; % 设置两侧 Y 轴刻度标签字体大小
end

% 添加图例
if isempty(h2)
    legend(h1, {'Event'}, 'Location', 'best','FontSize', FontSize-5);
else
    legend([h1 h2], {'HPGe Event','Amplitude'}, 'Location', 'best','FontSize', FontSize-10);
end

% 添加标题和网格
title('', 'FontName', 'Times New Roman', 'FontSize', FontSize);
xlabel('Date', 'FontName', 'Times New Roman', 'FontSize', FontSize);
yyaxis left; ylabel('Energy (keV)', 'FontName', 'Times New Roman', 'FontSize', FontSize);
yyaxis right; ylabel('Amplitude (mm)', 'FontName', 'Times New Roman', 'FontSize', FontSize);
grid off; % 添加网格
hold off;

%% 保存图像

% 调整图形大小和分辨率
set(gcf, 'Units', 'Inches', 'Position', [0 0 19.2 10.8]); % 设置图形窗口大小为 1920x1080
set(gcf, 'Color', 'w'); % 设置背景颜色为白色
% 保存图片
filename = sprintf('../images/scatter_plot_%s_to_%s_energy_%d_to_%d.png', ...
    datestr(starttime, 'yyyy-mm-dd_HH-MM-SS'), ...
    datestr(endtime, 'yyyy-mm-dd_HH-MM-SS'), ...
    energyrange(1), energyrange(2));
print(gcf, filename, '-dpng', '-r300'); % 保存为 PNG，分辨率为 300 ppi

%% 保存变量

% 定义保存路径
savePath = '../data'; % 相对路径，表示当前目录的上一级目录中的 data 文件夹
% 确保保存路径存在
if ~exist(savePath, 'dir')
    mkdir(savePath); % 如果目录不存在，则创建目录
end
% 定义保存的文件名
saveFileName = fullfile(savePath, 'HPGe_signal.mat');
% 检查工作区中是否存在 HPGe_signal 变量
if exist('HPGe_signal', 'var')
    % 保存 HPGe_signal 到指定路径
    save(saveFileName, 'HPGe_signal');
    disp(['HPGe_signal 已成功保存到 ', saveFileName]);
else
    % 如果变量不存在，提示用户
    warning('工作区中不存在 HPGe_signal 变量。');
end

clear ax filteredIdx endtime starttime energyrange filename FontSize data_transformed saveFileName savePath

