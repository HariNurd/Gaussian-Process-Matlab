%% Membaca file CSV
clc, format compact

data = readtable("Data 4.csv");

%% Membuat array untuk tanggal
dates = datetime(2001, 1, 1):calmonths(1):datetime(2020, 12, 1); % Adding dates to the data
dates = sort(dates); % Sort dates
dates = dates(:);

%% Membuat tabel untuk kolom tanggal
data = addvars(data, dates, 'NewVariableNames', 'Date', 'Before', 1);

%% Hapus kolom yang tidak dipakai
data = removevars(data, 'Tanggal');

%% Bagi data menjadi data train and data test
X = data(:, 2:end-2); % Exclude the first (Tanggal) and last (Hotspots) columns for features
y = data.('JumlahTitikApi'); % Target variable

% Hitung baris yang dipakai (80% dari total baris)
numRows = size(X, 1);
splitIndex = floor(0.8 * numRows);

% Pilih 80% pertama dari seluruh baris data
XTrain = X(1:splitIndex, :);
yTrain = y(1:splitIndex, :);
XData_Train = data(1:splitIndex, :);

% pilih 20% terakhir dari baris data
XTest = X((splitIndex+1):end, :);
yTest = y((splitIndex+1):end, :);
XData_Test = data((splitIndex+1):end, :);


%% Membuat dan melatih model

% Model awal (tanpa nilai awal)
rng(0,'twister') % for reproducibility
gprMdl_awal0 = fitrgp(XTrain, yTrain,'KernelFunction','squaredexponential','Standardize',true, 'FitMethod','exact', ...
    'PredictMethod','exact','BasisFunction','constant', 'ActiveSetMethod','likelihood');

rng(0,'twister')
[ypred_awal0, ydev_awal0, yint_awal0] = predict(gprMdl_awal0, XTest);
rng(0,'twister')
[ypred_train_awal0, ydev_train_awal0, yint_train_awal0] = resubPredict(gprMdl_awal0);

%% Metrik akurasi model awal
format shortG
% hitung nilai MSE and RMSE
mse_awal0_train = resubLoss(gprMdl_awal0);
mse_awal0_test = loss(gprMdl_awal0,XTest,yTest);
rmse_awal0_train = sqrt(mse_awal0_train)
rmse_awal0_test = sqrt(mse_awal0_test)
% Calculate MAE
mae_awal0_train = mean(abs(ypred_train_awal0 - yTrain));
mae_awal0_test = mean(abs(ypred_awal0 - yTest));
% Calculate R-squared all
R2_awal0_train = 1 - ((sum((yTrain - ypred_train_awal0).^2)) / (sum((yTrain - mean(yTrain)).^2)));
R2_awal0_test = 1 - ((sum((yTest - ypred_awal0).^2)) / (sum((yTest - mean(yTest)).^2)));

%% Pemilihan fungsi kernel
% Squared Exponential
rng(0,'twister') % for reproducibility
gprMdl_awal_SE = fitrgp(XTrain, yTrain,'KernelFunction','squaredexponential','Standardize',true, 'FitMethod','exact', ...
    'PredictMethod','exact','BasisFunction','constant', 'ActiveSetMethod','likelihood');

rng(0,'twister')
[ypred_awal_SE, ydev_awal_SE, yint_awal_SE] = predict(gprMdl_awal_SE, XTest);
rng(0,'twister')
[ypred_train_awal_SE, ydev_train_awal_SE, yint_train_awal_SE] = resubPredict(gprMdl_awal_SE);

% Matern32
rng(0,'twister') % for reproducibility
gprMdl_awal_32 = fitrgp(XTrain, yTrain,'KernelFunction','matern32','Standardize',true, 'FitMethod','exact', ...
'PredictMethod','exact','BasisFunction','constant', 'ActiveSetMethod','likelihood');

rng(0,'twister')
[ypred_awal_32, ydev_awal_32, yint_awal_32] = predict(gprMdl_awal_32, XTest);
rng(0,'twister')
[ypred_train_awal_32, ydev_train_awal_32, yint_train_awal_32] = resubPredict(gprMdl_awal_32);

% ARD Squared Exponential
rng(0,'twister') % for reproducibility
gprMdl_awal_ardSE = fitrgp(XTrain, yTrain,'KernelFunction','ardsquaredexponential','Standardize',true, 'FitMethod','exact', ...
    'PredictMethod','exact','BasisFunction','constant', 'ActiveSetMethod','likelihood');

rng(0,'twister')
[ypred_awal_ardSE, ydev_awal_ardSE, yint_awal_ardSE] = predict(gprMdl_awal_ardSE, XTest);
rng(0,'twister')
[ypred_train_awal_ardSE, ydev_train_awal_ardSE, yint_train_awal_ardSE] = resubPredict(gprMdl_awal_ardSE);

% Ard Matern32
rng(0,'twister') % for reproducibility
gprMdl_awal_ard32 = fitrgp(XTrain, yTrain,'KernelFunction','ardmatern32','Standardize',true, 'FitMethod','exact', ...
'PredictMethod','exact','BasisFunction','constant', 'ActiveSetMethod','likelihood');

rng(0,'twister')
[ypred_awal_ard32, ydev_awal_ard32, yint_awal_ard32] = predict(gprMdl_awal_ard32, XTest);
rng(0,'twister')
[ypred_train_awal_ard32, ydev_train_awal_ard32, yint_train_awal_ard32] = resubPredict(gprMdl_awal_ard32);

%% Metrik akurasi pemilihan kernel
format shortG
% Hitung nilai RMSE semua kernel
% SE
mse_awalSE_train = resubLoss(gprMdl_awal_SE);
mse_awalSE_test = loss(gprMdl_awal_SE,XTest,yTest);
rmse_awalSE_train = sqrt(mse_awalSE_train)
rmse_awalSE_test = sqrt(mse_awalSE_test)
% ardSE
mse_awalardSE_train = resubLoss(gprMdl_awal_ardSE);
mse_awalardSE_test = loss(gprMdl_awal_ardSE,XTest,yTest);
rmse_awalardSE_train = sqrt(mse_awalardSE_train)
rmse_awalardSE_test = sqrt(mse_awalardSE_test)
% Matern32
mse_awal32_train = resubLoss(gprMdl_awal_32);
mse_awal32_test = loss(gprMdl_awal_32,XTest,yTest);
rmse_awal32_train = sqrt(mse_awal32_train)
rmse_awal32_test = sqrt(mse_awal32_test)
% ardmatern32
mse_awalard32_train = resubLoss(gprMdl_awal_ard32);
mse_awalard32_test = loss(gprMdl_awal_ard32,XTest,yTest);
rmse_awalard32_train = sqrt(mse_awalard32_train)
rmse_awalard32_test = sqrt(mse_awalard32_test)

% Hitung nilai MAE semua kernel
% SE
mae_awalSE_train = mean(abs(ypred_train_awal_SE - yTrain));
mae_awalSE_test = mean(abs(ypred_awal_SE - yTest));
% ArdSE
mae_awalardSE_train = mean(abs(ypred_train_awal_ardSE - yTrain));
mae_awalardSE_test = mean(abs(ypred_awal_ardSE - yTest));
% Matern32
mae_awal32_train = mean(abs(ypred_train_awal_32 - yTrain));
mae_awal32_test = mean(abs(ypred_awal_32 - yTest));
% ArdMatern32
mae_awalard32_train = mean(abs(ypred_train_awal_ard32 - yTrain));
mae_awalard32_test = mean(abs(ypred_awal_ard32 - yTest));

% Hitung nilai R-squared semua kernel
R2_awalSE_train = 1 - ((sum((yTrain - ypred_train_awal_SE).^2)) / (sum((yTrain - mean(yTrain)).^2)));
R2_awalSE_test = 1 - ((sum((yTest - ypred_awal_SE).^2)) / (sum((yTest - mean(yTest)).^2)));
R2_awalardSE_train = 1 - ((sum((yTrain - ypred_train_awal_ardSE).^2)) / (sum((yTrain - mean(yTrain)).^2)));
R2_awalardSE_test = 1 - ((sum((yTest - ypred_awal_ardSE).^2)) / (sum((yTest - mean(yTest)).^2)));
R2_awal32_train = 1 - ((sum((yTrain - ypred_train_awal_32).^2)) / (sum((yTrain - mean(yTrain)).^2)));
R2_awal32_test = 1 - ((sum((yTest - ypred_awal_32).^2)) / (sum((yTest - mean(yTest)).^2)));
R2_awalard32_train = 1 - ((sum((yTrain - ypred_train_awal_ard32).^2)) / (sum((yTrain - mean(yTrain)).^2)));
R2_awalard32_test = 1 - ((sum((yTest - ypred_awal_ard32).^2)) / (sum((yTest - mean(yTest)).^2)));

%% Model awal (dengan nilai awal)

% Nilai awal parameter
sigma0 = std(yTrain);
sigmaF0 = sigma0;
d = size(XTrain,2);
sigmaM0 = 10*ones(d,1);
Theta0 = [sigmaM0; sigmaF0];

rng(0,'twister') % for reproducibility
gprMdl_awal1 = fitrgp(XTrain, yTrain,'KernelFunction','ardsquaredexponential','Standardize',true,'Sigma',sigmaF0,'KernelParameters',Theta0,'BasisFunction','constant', ...
    'FitMethod','exact', 'PredictMethod','exact','ActiveSetMethod','likelihood');

rng(0,'twister')
[ypred_awal1, ydev_awal1, yint_awal1] = predict(gprMdl_awal1, XTest);
rng(0,'twister')
[ypred_train_awal1, ydev_train_awal1, yint_train_awal1] = resubPredict(gprMdl_awal1);

%% Metrik akurasi dengan nilai awal
format shortG
% Hitung nilai RMSE
mse_awal1_train = resubLoss(gprMdl_awal1);
mse_awal1_test = loss(gprMdl_awal1,XTest,yTest);
rmse_awal1_train = sqrt(mse_awal1_train)
rmse_awal1_test = sqrt(mse_awal1_test)
% Hitung nilai MAE
mae_awal1_train = mean(abs(ypred_train_awal1 - yTrain));
mae_awal1_test = mean(abs(ypred_awal1 - yTest));
% Hitung nilai R-squared semua kernel
R2_awal1_train = 1 - ((sum((yTrain - ypred_train_awal1).^2)) / (sum((yTrain - mean(yTrain)).^2)));
R2_awal1_test = 1 - ((sum((yTest - ypred_awal1).^2)) / (sum((yTest - mean(yTest)).^2)));

%% Model dengan Bayesian optimization
rng(0,'twister') % for reproducibility
gprMdl_bayes = fitrgp(X, y,'KernelFunction','ardsquaredexponential','BasisFunction','constant', ...
    'OptimizeHyperparameters',{'Sigma','Standardize'},'HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',30, ...
    'AcquisitionFunctionName','expected-improvement-plus','optimizer', 'bayesopt','kfold',20));


%% Model dengan grid search
rng(0,'twister') % for reproducibility
gprMdl_grid = fitrgp(X, y,'KernelFunction','ardsquaredexponential','BasisFunction','constant', ...
    'OptimizeHyperparameters',{'Sigma','Standardize'},'HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',30, ...
    'optimizer', 'gridsearch','kfold',20));

%% Model dengan random search
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
%% Plot #2
figure();

hold on
plot(XData_Train.Date, XData_Train.JumlahTitikApi,'b');
plot(XData_Train.Date, ypred_train_awal0, 'r');
plot(XData_Test.Date, XData_Test.JumlahTitikApi,'b');
plot(XData_Test.Date, ypred_awal0, 'r');
xline(5835, '--k', 'DisplayName', 'Train/Test Boundary')
grid on
xlabel('Periode');
ylabel('Jumlah Hotspots');
hold off
title('Model Regresi awal GP data training & data testing');
% legend('show');
legend({'Data Aktual','Prediksi'},'Location','Best');

%% Plot #3
figure();

hold on
plot(XData_Train.Date, XData_Train.JumlahTitikApi,'b');
plot(XData_Train.Date, ypred_all_bayes, 'r');
grid on
xlabel('Periode');
ylabel('Jumlah Hotspots');
hold off
title('Model Regresi GP menggunakan Bayesian optimization');
% legend('show');
legend({'Data Aktual','Prediksi'},'Location','Best');

%% Plot #4
figure();

hold on
plot(XData_Train.Date, XData_Train.JumlahTitikApi,'b');
plot(XData_Train.Date, ypred_all_grid, 'r');
grid on
xlabel('Periode');
ylabel('Jumlah Hotspots');
hold off
title('Model Regresi GP menggunakan grid search');
% legend('show');
legend({'Data Aktual','Prediksi'},'Location','Best');

%% Plot #5
figure();

hold on
plot(XData_Train.Date, XData_Train.JumlahTitikApi,'b');
plot(XData_Train.Date, ypred_all_random, 'r');
grid on
xlabel('Periode');
ylabel('Jumlah Hotspots');
hold off
title('Model Regresi GP menggunakan random search');
% legend('show');
legend({'Data Aktual','Prediksi'},'Location','Best');

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
% gabung data menjadi matriks 3x1
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
