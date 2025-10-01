function [time_sec, data, filePrefix] = load_standard_csv(csvPath)
% 读取标准化 CSV：time_sec, channel_1, channel_2, channel_3
T = readtable(csvPath);
reqCols = {'time_sec','channel_1','channel_2','channel_3'};
assert(all(ismember(reqCols, T.Properties.VariableNames)), ...
    'CSV 缺少所需列：%s', strjoin(setdiff(reqCols, T.Properties.VariableNames), ', '));

time_sec = T.time_sec(:);
data = [T.channel_1(:), T.channel_2(:), T.channel_3(:)];

[~, name, ~] = fileparts(csvPath);
filePrefix = name;
end
