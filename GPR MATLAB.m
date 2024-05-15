%% Read CSV
clc, format compact

data = readtable("Data 4.csv");

%% Create an array of dates
dates = datetime(2001, 1, 1):calmonths(1):datetime(2020, 12, 1); % Adding dates to the data
dates = sort(dates); % Sort dates
dates = dates(:);

%% Create a table with the date column
data = addvars(data, dates, 'NewVariableNames', 'Date', 'Before', 1);

%% Remove unnecessary feature
data = removevars(data, 'Tanggal');

%% Split data train and data test
X = data(:, 2:end-2); % Exclude the first (Tanggal) and last (Hotspots) columns for features
y = data.('JumlahTitikApi'); % Target variable

% Calculate the number of rows to select (80% of total rows)
numRows = size(X, 1);
splitIndex = floor(0.8 * numRows);

% Select the first 80% of the rows
XTrain = X(1:splitIndex, :);
yTrain = y(1:splitIndex, :);
XData_Train = data(1:splitIndex, :);

% Select the last 20% of the rows
XTest = X((splitIndex+1):end, :);
yTest = y((splitIndex+1):end, :);
XData_Test = data((splitIndex+1):end, :);


%% Model default
rng(0,'twister') % for reproducibility
gprMdl_bayes = fitrgp(X, y,'KernelFunction','ardsquaredexponential','BasisFunction','constant', ...
    'OptimizeHyperparameters',{'Sigma','Standardize'},'HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',30, ...
    'AcquisitionFunctionName','expected-improvement-plus','optimizer', 'bayesopt','kfold',20));
rng(0,'twister') % for reproducibility
gprMdl_grid = fitrgp(X, y,'KernelFunction','ardsquaredexponential','BasisFunction','constant', ...
    'OptimizeHyperparameters',{'Sigma','Standardize'},'HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',30, ...
    'optimizer', 'gridsearch','kfold',20));
rng(0,'twister') % for reproducibility
gprMdl_random = fitrgp(X, y,'KernelFunction','ardsquaredexponential','BasisFunction','constant', ...
    'OptimizeHyperparameters',{'Sigma','Standardize'},'HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',30, ...
    'optimizer', 'randomsearch','kfold',20));
%%
rng(0,'twister')
[ypred_all_bayes, ydev_all_bayes, yint_all_bayes] = predict(gprMdl_bayes, X);
rng(0,'twister')
[ypred_all_grid, ydev_all_grid, yint_all_grid] = predict(gprMdl_grid, X);
rng(0,'twister')
[ypred_all_random, ydev_all_ranom, yint_all_random] = predict(gprMdl_random, X);
%%
format shortG
% Calculate MSE and RMSE
mse_bayes = loss(gprMdl_bayes,X,y);
mse_grid = loss(gprMdl_grid,X,y);
mse_random = loss(gprMdl_random,X,y);
rmse_bayes = sqrt(mse_bayes)
rmse_grid = sqrt(mse_grid)
rmse_random = sqrt(mse_random)
% Calculate MAE
mae_bayes = mean(abs(ypred_all_bayes - y))
mae_grid = mean(abs(ypred_all_grid - y))
mae_random = mean(abs(ypred_all_random - y))
% Calculate R-squared all
R2_bayes = 1 - ((sum((y - ypred_all_bayes).^2)) / (sum((y - mean(y)).^2)))
R2_grid = 1 - ((sum((y - ypred_all_grid).^2)) / (sum((y - mean(y)).^2)))
R2_random = 1 - ((sum((y - ypred_all_random).^2)) / (sum((y - mean(y)).^2)))

%% Plot #1
figure();
tiledlayout("vertical");

nexttile
hold on
plot(data.Date, data.JumlahTitikApi,'b');
plot(data.Date, ypred_all_bayes, 'r');
plot(data.Date, ypred_all_grid, 'y');
ylabel('Jumlah Hotspot');
grid on
hold off
legend({'Data Aktual','Bayes Opt','Grid Search'},'Location','Best');

nexttile
hold on
plot(data.Date, data.JumlahTitikApi,'b');
plot(data.Date, ypred_all_grid, 'y');
plot(data.Date, ypred_all_random, 'g');
ylabel('Jumlah Hotspot');
grid on
hold off
legend({'Data Aktual','Grid Search','Random Search'},'Location','Best');

nexttile
hold on
plot(data.Date, data.JumlahTitikApi,'b');
plot(data.Date, ypred_all_bayes, 'r');
plot(data.Date, ypred_all_random, 'g');
xlabel('Periode (Tahun)');
ylabel('Jumlah Hotspot');
grid on
hold off
legend({'Data Aktual','Bayes Opt','Random Search'},'Location','Best');
%% Plot #1
figure();

hold on
plot(XData_Train.Date, XData_Train.JumlahTitikApi,'b');
plot(XData_Train.Date, ypred_train_awal, 'r');
plot(XData_Test.Date, XData_Test.JumlahTitikApi,'b');
plot(XData_Test.Date, ypred_bayes, 'r');
xline(5835, '--k', 'DisplayName', 'Train/Test Boundary')
grid on
xlabel('Periode');
ylabel('Jumlah Hotspots');
hold off
title('Model Regresi GP data training & data testing');
% legend('show');
legend({'Data Aktual','Prediksi'},'Location','Best');

%% Initial values of the kernel parameters
sigma0 = std(yTrain);
sigmaF0 = sigma0;
sigmaF = 1296.4;
d = size(XTrain,2);
sigmaM0 = 10*ones(d,1);
Theta0 = [sigmaM0; sigmaF0];
Theta = [19486, 1.103e+05, 15.923,  1.1506e+05,  1.4305e+06, 3498];
Beta = 3114.4;
opts = statset('fitrgp');
opts.TolFun = 1e-2;

%%
gprMdl.KernelInformation
gprMdl.KernelInformation.KernelParameterNames
gprMdl.KernelInformation.KernelParameters
gprMdl.Sigma

%%
sigmaM = gprMdl_bayes.KernelInformation.KernelParameters(1:end-1,1);
figure()
plot((1:d)',log(sigmaM),'ro-');
xlabel('Length scale number');
ylabel('Log of length scale');

%%
nRows = size(data, 1);
aktual = y((nRows-11):nRows, :);
bayes20 = ypred_all_bayes((nRows-11):nRows, :);
grid20 = ypred_all_grid((nRows-11):nRows, :);
random20 = ypred_all_random((nRows-11):nRows, :);
%%
% Menginisialisasi variabel untuk menyimpan RMSE per bulan
rmse_per_month = zeros(12, 1);

% Menghitung RMSE untuk setiap bulan
for month = 1:12
    month_indices = month:12:240;  % Indeks untuk bulan tertentu setiap tahun
    month_actual = y(month_indices);
    month_predicted_bayes = ypred_all_bayes(month_indices);
    month_predicted_grid = ypred_all_grid(month_indices);
    month_predicted_random = ypred_all_random(month_indices);
    
    % Menghitung RMSE untuk bulan ini
    rmse_per_month_bayes(month) = sqrt(mean((month_actual - month_predicted_bayes).^2));
    rmse_per_month_grid(month) = sqrt(mean((month_actual - month_predicted_grid).^2));
    rmse_per_month_random(month) = sqrt(mean((month_actual - month_predicted_random).^2));

end

%%
% Menyusun data RMSE dalam matriks
rmse_data = [rmse_per_month_bayes rmse_per_month_grid rmse_per_month_random];

% Membuat bar chart
figure; % Membuka figure baru
bar(rmse_data, 'grouped'); % Membuat bar chart dengan grup
title('Perbandingan RMSE per Bulan untuk Tiga Metode');
xlabel('Bulan');
ylabel('RMSE');
legend('Bayes', 'Grid', 'Random'); % Menambahkan legenda untuk membedakan metode

% Mengatur ticks pada x-axis untuk menampilkan nama bulan
xticks(1:12);
xticklabels({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'});


%%
% Data contoh, gantikan dengan data RMSE Anda
rmse_data = [rmse_per_month_bayes; rmse_per_month_grid; rmse_per_month_random];

% Membuat figure dan bar chart
figure;
set(gcf, 'Position', [100, 100, 1200, 600]); % Mengubah ukuran window figure (x, y, width, height)
b = bar(rmse_data', 'grouped');
hold on; % Menahan figure agar tidak tertutup oleh perintah berikutnya

% Menambahkan garis vertikal sebagai pemisah
for i = 1.5:1:11.5
    xline(i, '--k', 'LineWidth', 1);
end

% Judul dan label
title('Perbandingan RMSE per Bulan untuk Tiga Metode');
xlabel('Bulan');
ylabel('RMSE');
legend('Bayes', 'Grid', 'Random', 'Location', 'northwest');

% Mengatur ticks pada x-axis untuk menampilkan nama bulan
xticks(1:12);
xticklabels({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'});

% Menambahkan label teks ke setiap bar
for i = 1:12
    for j = 1:length(b)
        x = b(j).XData(i) + b(j).XOffset; % Posisi x untuk label
        y = b(j).YData(i); % Posisi y untuk label
        %text(x, y, sprintf('%.2f', y), 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'FontSize', 8, 'Rotation', 90);
    end
end

hold off; % Melepaskan hold
