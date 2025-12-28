clear;
% 定义文件夹路径
folderPath = '../data/VibrationSensor_txt/';
savePath = '../data';

% 检查保存路径是否存在，如果不存在则创建
if ~exist(savePath, 'dir')
    mkdir(savePath);
end

% 获取目录下符合格式的 .txt 文件（detector_数字_日期.txt）
fileList = dir(fullfile(folderPath, 'detector_*_*.txt'));
% 初始化日期和 detectornumber
startDate = datetime('2099-12-31'); % 初始化为一个很大的日期
endDate = datetime('1900-01-01'); % 初始化为一个很小的日期
detectornumbers = []; % 用于存储所有 detectornumber

% 遍历文件列表，提取日期和 detectornumber
for i = 1:length(fileList)
    fileName = fileList(i).name;
    % 提取文件名中的日期和 detectornumber
    parts = strsplit(fileName, '_');
    detectornumber = str2double(parts{2}); % 提取 detectornumber
    dateStr = parts{3};
    dateStr = strrep(dateStr, '.txt', ''); % 去掉文件扩展名
    fileDate = datetime(dateStr, 'InputFormat', 'yyyy-MM-dd');
    
    % 更新 startDate 和 endDate
    if fileDate < startDate
        startDate = fileDate;
    end
    if fileDate > endDate
        endDate = fileDate;
    end
    
    % 将当前 detectornumber 添加到数组中
    detectornumbers = [detectornumbers, detectornumber];
end

% 如果没有找到任何文件，提示用户并退出
if startDate == datetime('2099-12-31') || endDate == datetime('1900-01-01')
    warning('没有找到任何有效的 .txt 文件。');
    return;
end

% 找到最大的 detectornumber
detectornumber = max(detectornumbers);

% 初始化一个 cell 来存储结果
results = cell(detectornumber, days(endDate - startDate) + 1);

% 遍历每个 detectornumber
for detectorIdx = 1:detectornumber
    % 使用 parfor 并行处理每个日期
    parfor dateIdx = 0:days(endDate - startDate)
        date = startDate + days(dateIdx);
        % 生成文件名
        fileName = sprintf('detector_%d_%d-%02d-%02d.txt', detectorIdx, date.Year, date.Month, date.Day);
        filePath = fullfile(folderPath, fileName);
        
        % 检查对应的 .mat 文件是否存在
        matFileName = sprintf('detector%d%d%02d%02d.mat', detectorIdx, date.Year, date.Month, date.Day);
        matFilePath = fullfile(savePath, matFileName);
        
        if exist(matFilePath, 'file') % 如果 .mat 文件存在，跳过读取 .txt 文件
            warning('文件 %s 已存在，跳过处理。', matFilePath);
            continue; % 跳过当前循环
        end

        % 检查文件是否存在
        if ~exist(filePath, 'file')
            warning('文件 %s 不存在，跳过处理。', fileName);
            continue; % 跳过当前循环
        end

        % 设置导入选项
        opts = delimitedTextImportOptions("NumVariables", 15);
        % 指定范围和分隔符
        opts.DataLines = [1, Inf];
        opts.Delimiter = ",";
        % 指定列名称和类型
        opts.VariableNames = ["VarName1", "VarName2", "VarName3", "VarName4", "VarName5", "VarName6", ...
                              "VarName7", "VarName8", "VarName9", "VarName10", "VarName11", "VarName12", ...
                              "VarName13", "VarName14", "VarName15"];
        opts.VariableTypes = ["datetime", "double", "double", "double", "double", "double", ...
                              "double", "double", "double", "double", "double", "double", ...
                              "double", "double", "string"];
        % 指定文件级属性
        opts.ExtraColumnsRule = "ignore";
        opts.EmptyLineRule = "read";
        % 指定变量属性
        opts = setvartype(opts, "VarName1", "datetime");
        opts = setvaropts(opts, "VarName15", "WhitespaceRule", "preserve");
        opts = setvaropts(opts, "VarName15", "EmptyFieldRule", "auto");
        opts = setvaropts(opts, "VarName1", "InputFormat", "yyyy-MM-dd HH:mm:ss.SSS", "DatetimeFormat", "preserveinput");
        % 导入数据
        tableVar = readtable(filePath, opts);
        % 清理时间一列的日期。有时候年份前面会多个2
        % 将时间数据转换为字符串数组
        timeStrings = string(tableVar.VarName1);
        % 提取时间部分（从第12个字符开始到结尾）
        timeParts = extractAfter(timeStrings, ' ');
        % 组合成新的日期时间字符串
        fixedTimeStrings = strcat(string(date.Year) + "-" + string(date.Month) + "-" + string(date.Day) + " ", timeParts);
        % 转换回datetime类型
        tableVar.VarName1 = datetime(fixedTimeStrings, 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');
        
        % 将结果存储到 cell 中
        results{detectorIdx, dateIdx + 1} = tableVar;
    end
end

% 将保存结果到工作区
for detectorIdx = 1:detectornumber
    for dateIdx = 0:days(endDate - startDate)
        if ~isempty(results{detectorIdx, dateIdx + 1})
            varName = sprintf('detector%d%d%02d%02d', detectorIdx, results{detectorIdx, dateIdx + 1}.VarName1(1).Year, ...
                              results{detectorIdx, dateIdx + 1}.VarName1(1).Month, results{detectorIdx, dateIdx + 1}.VarName1(1).Day);
            assignin('base', varName, results{detectorIdx, dateIdx + 1});
        end
    end
end
% 清理临时变量
clear folderPath fileList detectornumbers dateStr detectorIdx

%% table转换成double

% 循环处理每个日期
for detectorIdx = 1:detectornumber
    for date = startDate:endDate
        varName = sprintf('detector%d%d%02d%02d',detectorIdx, date.Year, date.Month, date.Day);
        % 检查变量是否存在
        if ~exist(varName, 'var')
            continue; % 如果变量不存在，跳过处理
        end
        file=eval(varName);
        file(:, 15) = [];
        timeStrings = file.VarName1;
        if ~strcmp(timeStrings(1), date)
            % 如果日期不匹配，删除第一列
            file.VarName1(:, 1) = [];
        end
        timeFormat = 'yyyy-MM-dd HH:mm:ss.SSS'; % 根据你的数据格式调整
        timeValues = datetime(timeStrings, 'InputFormat', timeFormat);
        secondsSinceEpoch = seconds(timeValues - datetime('1970-01-01'));
        file.VarName1 = secondsSinceEpoch;
        % 将处理后的 file 转换回矩阵，并存储到原变量名
        eval([varName ' = table2array(file);']);  % 关键修改：动态变量名赋值
        clear timeStrings timeFormat timeValues secondsSinceEpoch file
    end
end

clear detectorIdx endDate startDate varName date 

%% 获取当前工作区的所有变量名
varNames = who;
% 遍历变量并保存
for i = 1:length(varNames)
    varName = varNames{i};
    % 检查变量名是否包含 "detector"
    if contains(varName, 'detector')
        % 检查对应的 .mat 文件是否存在
        matFilePath = fullfile(savePath, [varName '.mat']);
        if exist(matFilePath, 'file') % 如果 .mat 文件存在，跳过保存
            warning('文件 %s 已存在，跳过保存。', matFilePath);
            continue;
        end
        save(fullfile(savePath, [varName '.mat']), varName);
    end
end
disp('所有包含 "detector" 的变量已保存到指定路径。');

