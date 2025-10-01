function make_chinese()
% 设置中文字体（按优先级回退）
try
    set(groot, 'defaultAxesFontName', 'Microsoft YaHei');
    set(groot, 'defaultTextFontName', 'Microsoft YaHei');
catch
    try
        set(groot, 'defaultAxesFontName', 'SimHei');
        set(groot, 'defaultTextFontName', 'SimHei');
    catch
        % 使用系统默认
    end
end
set(groot, 'defaultAxesFontSize', 12);
end
