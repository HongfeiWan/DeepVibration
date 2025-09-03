%% 导入数据
% 定义文件路径
folderPath = '../data/';
matFilePath = fullfile(folderPath, 'HPGe_signal.mat');
mFilePath = fullfile('./', 'HPGe_detector_Data_Viewer.m');

%% 检查 HPGe_signal.mat 文件是否存在
if exist(matFilePath, 'file')
    % 如果文件存在，加载 .mat 文件
    load(matFilePath);
    disp('HPGe_signal.mat 文件已加载。');
else
    % 如果文件不存在，检查 HPGe_detector_Data_Viewer.m 文件是否存在
    if exist(mFilePath, 'file')
        % 如果脚本文件存在，运行该脚本
        run(mFilePath);
        disp('HPGe_signal.mat 文件不存在，已运行 HPGe_detector_Data_Viewer.m 脚本。');
    else
        % 如果脚本文件也不存在，提示用户
        error('HPGe_signal.mat 文件和 HPGe_detector_Data_Viewer.m 脚本都不存在。');
    end
end
clear folderPath matFilePath mFilePath

%% 时间转换
epochStart = datetime('1970-01-01', 'TimeZone', '');
data_transformed = epochStart + seconds(HPGe_signal.Time);
data_transformed.Format = 'yyyy-MM-dd HH:mm:ss.SSS';
clear epochStart

%% 分bin
% 假设 HPGe_signal.Time 是数值型数据（秒），HPGe_signal.Value 是能量值
% 假设 data_transformed 是 datetime 类型的向量
% 定义时间 bin 的大小（例如，每 600 秒一个 bin，即 10 分钟）
timeBinSize = 600; % 600 秒（10 分钟）

% 定义能量范围
energyRange = [0, 0.5]; % 0-0.5 keV

% 获取时间范围
startTime = min(data_transformed);
endTime = max(data_transformed);

% 创建时间 bin
timeBins = startTime:seconds(timeBinSize):endTime;

% 初始化计数数组
counts = zeros(length(timeBins) - 1, 1);

% 遍历每个时间 bin，计算满足能量范围的计数
parfor i = 1:length(timeBins) - 1
    % 当前时间 bin 的范围
    binStart = timeBins(i);
    binEnd = timeBins(i + 1);
    % 找到当前时间 bin 内的事件
    inBinIdx = data_transformed >= binStart & data_transformed < binEnd;
    % 找到满足能量范围的事件
    inEnergyRangeIdx = HPGe_signal.Value >= energyRange(1) & HPGe_signal.Value <= energyRange(2);
    % 计算当前时间 bin 内满足能量范围的计数
    counts(i) = sum(inBinIdx & inEnergyRangeIdx);
end



% 绘制结果
figure;
bar(timeBins(1:end-1), counts, 'FaceColor', 'red');

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
% legendStr = sprintf('Event per %d Seconds', timeBinSize);
% legend(legendStr, 'Location', 'best');
% 添加标题和网格
title('', 'FontName', 'Times New Roman', 'FontSize', FontSize);
xlabel('Date', 'FontName', 'Times New Roman', 'FontSize', FontSize);
ylabel('Count', 'FontName', 'Times New Roman', 'FontSize', FontSize);
grid off; % 添加网格
hold off;

% 调整图形大小和分辨率
set(gcf, 'Units', 'Inches', 'Position', [0 0 19.2 10.8]); % 设置图形窗口大小为 1920x1080
set(gcf, 'Color', 'w'); % 设置背景颜色为白色
% 保存图片
filename = sprintf('../images/%d_seconds_bar_plot_%s_to_%s_energy_%d_to_%d.png', ...
    timeBinSize, ...
    datestr(startTime, 'yyyy-mm-dd_HH-MM-SS'), ...
    datestr(endTime, 'yyyy-mm-dd_HH-MM-SS'), ...
    energyRange(1), energyRange(2));
print(gcf, filename, '-dpng', '-r300'); % 保存为 PNG，分辨率为 300 ppi

clear ax binEnd binStart counts data_transformed endTime energyRange filename FontSize i inBinIdx inEnergyRangeIdx legendStr startTime timeBins timeBinSize