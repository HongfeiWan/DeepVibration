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
starttime = datetime('2025-05-21 00:00:00.000', 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');
endtime = datetime('2025-05-30 00:00:00.000', 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');
energyrange=[0,1];%0-0.5keV
filteredIdx = (HPGe_signal.Value > energyrange(1) & HPGe_signal.Value < energyrange(2)) & ...
    (data_transformed >= starttime & data_transformed <= endtime);

% 画图
figure; % 创建新图形窗口
scatter(data_transformed(filteredIdx),HPGe_signal.Value(filteredIdx),'red','filled',SizeData=1)

% 设置坐标轴标签字体和字体大小
ax = gca; % 获取当前坐标轴对象
FontSize = 25;

xticklabels(datestr(ax.XAxis.TickValues, 'YYYY-mm-dd')); % 设置 X 轴刻度标签为 MM-dd 格式

ax.XLabel.FontName = 'Times New Roman'; % 设置 X 轴标签字体
ax.YLabel.FontName = 'Times New Roman'; % 设置 Y 轴标签字体
ax.XAxis.FontName = 'Times New Roman'; % 设置 X 轴刻度标签字体
ax.YAxis.FontName = 'Times New Roman'; % 设置 Y 轴刻度标签字体

ax.XLabel.FontSize = FontSize; % 设置 X 轴标签字体大小
ax.YLabel.FontSize = FontSize; % 设置 Y 轴标签字体大小
ax.XAxis.FontSize = FontSize-5; % 设置 X 轴刻度标签字体大小
ax.YAxis.FontSize = FontSize-5; % 设置 Y 轴刻度标签字体大小

% 添加图例
legend('Event', 'Location', 'best','FontSize', FontSize-5);

% 添加标题和网格
title('', 'FontName', 'Times New Roman', 'FontSize', FontSize);
xlabel('Date', 'FontName', 'Times New Roman', 'FontSize', FontSize);
ylabel('Energy (keV)', 'FontName', 'Times New Roman', 'FontSize', FontSize);
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

