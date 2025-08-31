%% Modeling for ZF43 titration

%% all summary plot, ZF43

% for only summary plot data
% reporter - 1
% ZF43 - 2
% bioreplicate - 3
% mGL-A_gmean - 4
% TagBFP-A_gmean - 5
% Fraction - 6
% Count - 7
% Normalized_mGL-A_gmean - 8
% Normalized_TagBFP-A_gmean - 9

% code starts here

filepath_noCre = '/Users/yunbeenbae/Documents/GitHub/Cre-loxP-ZF-promoters/flow_analysis_scripts/2024_promoter_editing_paper/datafiles/modeling_noCre_ZF43titration.xlsx';
filepath_Cre = '/Users/yunbeenbae/Documents/GitHub/Cre-loxP-ZF-promoters/flow_analysis_scripts/2024_promoter_editing_paper/datafiles/modeling_Cre_ZF43titration.xlsx';

data_noCre = readmatrix(filepath_noCre);
data_Cre = readmatrix(filepath_Cre);

x_zf_noCre = data_noCre(:,9); % TagBFP-A
y_rna_noCre = data_noCre(:,8); % mGL-A

x_zf_Cre = data_Cre(:,9); % TagBFP-A
y_rna_Cre = data_Cre(:,8); % mGL-A

%% removing burden points -- seems only necessary for ZF37?

% % gate visually
% zf_burden = 7; % selected upon visual inspection of graph
% positions = find(x_zf_Cre > zf_burden);
% 
% % remove the points related to burden
% x_zf_Cre(positions) = [];
% y_rna_Cre(positions) = [];

%% initializing TagBFP expression

x_zf_Cre = x_zf_Cre-x_zf_Cre(1);
x_zf_Cre(1) = 0.0000001; % to prevent problem with log transformations
x_zf_noCre = x_zf_noCre-x_zf_noCre(1);
x_zf_noCre(1) = 0.0000001; % to prevent problem with log transformations

%% setting basal expression

alpha_noCre = y_rna_noCre(1);
alpha_Cre = y_rna_Cre(1);

%% bootstrapping to generate parameters
% Define the number of bootstraps
num_bootstraps = 10000;

% Initialize arrays to store bootstrapped parameters
bootstrap_params_k_cat_noCre = zeros(num_bootstraps, 1);
bootstrap_params_K_M_noCre = zeros(num_bootstraps, 1);

bootstrap_params_k_cat_Cre = zeros(num_bootstraps, 1);
bootstrap_params_K_M_Cre = zeros(num_bootstraps, 1);

% Initialize matrices to store y-values
num_x_values = 100; % Number of x-values for interpolation
y_matrix_noCre = zeros(num_x_values, num_bootstraps);
y_matrix_Cre = zeros(num_x_values, num_bootstraps);

% Bootstrap resampling, curve fitting, and y-value generation for x_zf_noCre
for i = 1:num_bootstraps
    % Resample the data with replacement
    bootstrap_indices = randsample(length(x_zf_noCre), length(x_zf_noCre), true);
    x_zf_bootstrap = x_zf_noCre(bootstrap_indices);
    y_rna_bootstrap = y_rna_noCre(bootstrap_indices);

    % Fit the curve to the bootstrap sample
    objective_function = @(params, zf) f2(params, zf, alpha_noCre);
    params_guess = [1, 0.05];
    bootstrap_params = lsqcurvefit(objective_function, params_guess, x_zf_bootstrap, y_rna_bootstrap);

    % Store bootstrapped parameters
    bootstrap_params_k_cat_noCre(i) = bootstrap_params(1);
    bootstrap_params_K_M_noCre(i) = bootstrap_params(2);
    
    % Generate y values for logspace(x) for the current set of bootstrapped parameters
    x_values = logspace(log10(min(x_zf_noCre)), log10(max(x_zf_noCre)), num_x_values); % Define x values
    y_values = f2(bootstrap_params, x_values, alpha_noCre);
    y_matrix_noCre(:, i) = y_values;
end

% Bootstrap resampling, curve fitting, and y-value generation for x_zf_Cre
for i = 1:num_bootstraps
    % Resample the data with replacement
    bootstrap_indices = randsample(length(x_zf_Cre), length(x_zf_Cre), true);
    x_zf_bootstrap = x_zf_Cre(bootstrap_indices);
    y_rna_bootstrap = y_rna_Cre(bootstrap_indices);

    % Fit the curve to the bootstrap sample
    objective_function = @(params, zf) f2(params, zf, alpha_Cre);
    params_guess = [14, 0.2];
    bootstrap_params = lsqcurvefit(objective_function, params_guess, x_zf_bootstrap, y_rna_bootstrap);

    % Store bootstrapped parameters
    bootstrap_params_k_cat_Cre(i) = bootstrap_params(1);
    bootstrap_params_K_M_Cre(i) = bootstrap_params(2);
    
    % Generate y values for logspace(x) for the current set of bootstrapped parameters
    x_values = logspace(log10(min(x_zf_Cre)), log10(max(x_zf_Cre)), num_x_values); % Define x values
    y_values = f2(bootstrap_params, x_values, alpha_Cre);
    y_matrix_Cre(:, i) = y_values;
end

%% Plot histograms of bootstrapped parameters for x_zf_noCre
figure;
subplot(2, 2, 1);
histogram(bootstrap_params_k_cat_noCre, 'Normalization', 'probability');
title('Histogram of k_{cat} (No Cre)');
xlim([0,7]);

subplot(2, 2, 2);
histogram(bootstrap_params_K_M_noCre, 'Normalization', 'probability');
title('Histogram of K_{M} (No Cre)');
xlim([0,0.5]);

% Plot histograms of bootstrapped parameters for x_zf_Cre
subplot(2, 2, 3);
histogram(bootstrap_params_k_cat_Cre, 'Normalization', 'probability');
title('Histogram of k_{cat} (Cre)');
xlim([0,7]);

subplot(2, 2, 4);
histogram(bootstrap_params_K_M_Cre, 'Normalization', 'probability');
title('Histogram of K_{M} (Cre)');
xlim([0,0.5]);

%% exporting kcat and KM

csvwrite('kcat_noCre_ZF43.csv', bootstrap_params_k_cat_noCre);
csvwrite('kcat_Cre_ZF43.csv', bootstrap_params_k_cat_Cre);
csvwrite('KM_noCre_ZF43.csv', bootstrap_params_K_M_noCre);
csvwrite('KM_Cre_ZF43.csv', bootstrap_params_K_M_Cre);

%% % Plot y-values for x_zf_noCre
% figure;
% plot(x_values, y_matrix_noCre, 'Color', [0.8, 0.8, 0.8]); % Plot all bootstrapped curves
% xlabel('TagBFP-A Normalized by gmean of ZF43 0.125x');
% ylabel('mGL-A  Normalized by gmean of ZF43 0.125x');
% title('Bootstrapped Curves (No Cre)');
% 
% % Plot y-values for x_zf_Cre
% figure;
% plot(x_values, y_matrix_Cre, 'Color', [0.8, 0.8, 0.8]); % Plot all bootstrapped curves
% xlabel('TagBFP-A Normalized by gmean of ZF43 0.125x');
% ylabel('mGL-A  Normalized by gmean of ZF43 0.125x');
% title('Bootstrapped Curves (Cre)');

%% Plotting the 95% CI

% Calculate the mean of each row of the y_matrix for both conditions
mean_y_matrix_noCre = mean(y_matrix_noCre, 2);
mean_y_matrix_Cre = mean(y_matrix_Cre, 2);

% Sort y_matrix rows in ascending order
sorted_y_matrix_noCre = sort(y_matrix_noCre, 2);
sorted_y_matrix_Cre = sort(y_matrix_Cre, 2);

% Calculate the indices for the 2.5th and 97.5th percentiles
lower_percentile_index = round(0.025 * num_bootstraps);
upper_percentile_index = round(0.975 * num_bootstraps);

% Initialize arrays to store the 2.5% and 97.5% percentile values
lower_percentiles_noCre = zeros(num_x_values, 1);
upper_percentiles_noCre = zeros(num_x_values, 1);

lower_percentiles_Cre = zeros(num_x_values, 1);
upper_percentiles_Cre = zeros(num_x_values, 1);

% Calculate the 2.5% and 97.5% percentile values for x_zf_noCre
for i = 1:num_x_values
    lower_percentiles_noCre(i) = sorted_y_matrix_noCre(i, lower_percentile_index);
    upper_percentiles_noCre(i) = sorted_y_matrix_noCre(i, upper_percentile_index);
end

% Calculate the 2.5% and 97.5% percentile values for x_zf_Cre
for i = 1:num_x_values
    lower_percentiles_Cre(i) = sorted_y_matrix_Cre(i, lower_percentile_index);
    upper_percentiles_Cre(i) = sorted_y_matrix_Cre(i, upper_percentile_index);
end

% Plot the curves for x_zf_noCre
figure;
hold on;
plot(x_values, lower_percentiles_noCre, 'b', 'LineWidth', 1.5);
plot(x_values, upper_percentiles_noCre, 'b', 'LineWidth', 1.5);
% Shade the area between the two curves
fill([x_values, fliplr(x_values)], [lower_percentiles_noCre', fliplr(upper_percentiles_noCre')], 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
% Plot the mean as a dotted red line
plot(x_values, mean_y_matrix_noCre, 'r--', 'LineWidth', 1.5);
hold on;

% Plot the curves for x_zf_Cre
plot(x_values, lower_percentiles_Cre, 'g', 'LineWidth', 1.5);
plot(x_values, upper_percentiles_Cre, 'g', 'LineWidth', 1.5);
% Shade the area between the two curves
fill([x_values, fliplr(x_values)], [lower_percentiles_Cre', fliplr(upper_percentiles_Cre')], 'g', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
% Plot the mean as a dotted red line
plot(x_values, mean_y_matrix_Cre, 'k--', 'LineWidth', 1.5);

xlabel('TagBFP-A Normalized by gmean of ZF43 0.125x');
ylabel('mGL-A  Normalized by gmean of ZF43 0.125x');
title('Bootstrapped Curves for ZF43');
legend('2.5% Percentile (no Cre)', '97.5% Percentile (no Cre)','Shaded Area (no Cre)','Mean (no Cre)', ...
    '2.5% Percentile (Cre)', '97.5% Percentile (Cre)','Shaded Area (Cre)','Mean (Cre)');
hold off;

%% plotting parameters
% Sort bootstrapped k_cat and K_M values in ascending order
sorted_k_cat_noCre = sort(bootstrap_params_k_cat_noCre);
sorted_K_M_noCre = sort(bootstrap_params_K_M_noCre);

sorted_k_cat_Cre = sort(bootstrap_params_k_cat_Cre);
sorted_K_M_Cre = sort(bootstrap_params_K_M_Cre);

% Calculate the indices for the 2.5% and 97.5% percentiles
lower_percentile_index = round(0.025 * num_bootstraps);
upper_percentile_index = round(0.975 * num_bootstraps);

% Calculate the 2.5% and 97.5% percentiles for x_zf_noCre
lower_percentile_k_cat_noCre = sorted_k_cat_noCre(lower_percentile_index);
upper_percentile_k_cat_noCre = sorted_k_cat_noCre(upper_percentile_index);
lower_percentile_K_M_noCre = sorted_K_M_noCre(lower_percentile_index);
upper_percentile_K_M_noCre = sorted_K_M_noCre(upper_percentile_index);

% Calculate the 2.5% and 97.5% percentiles for x_zf_Cre
lower_percentile_k_cat_Cre = sorted_k_cat_Cre(lower_percentile_index);
upper_percentile_k_cat_Cre = sorted_k_cat_Cre(upper_percentile_index);
lower_percentile_K_M_Cre = sorted_K_M_Cre(lower_percentile_index);
upper_percentile_K_M_Cre = sorted_K_M_Cre(upper_percentile_index);

% Calculate the standard deviation for k_cat and K_M for both conditions
std_k_cat_noCre = std(bootstrap_params_k_cat_noCre);
std_K_M_noCre = std(bootstrap_params_K_M_noCre);
std_k_cat_Cre = std(bootstrap_params_k_cat_Cre);
std_K_M_Cre = std(bootstrap_params_K_M_Cre);

% Plot dot plot with error bars for k_cat for both conditions
figure;
hold on;
% range
% errorbar([1, 2], [lower_percentile_k_cat_noCre, lower_percentile_k_cat_Cre], [upper_percentile_k_cat_noCre, upper_percentile_k_cat_Cre], 'bo', 'LineWidth', 1.5);
% stdev
errorbar([1, 2], [mean(bootstrap_params_k_cat_noCre), mean(bootstrap_params_k_cat_Cre)], [std_k_cat_noCre, std_k_cat_Cre], 'bo', 'LineWidth', 1.5);
xlabel('Condition');
ylabel('k_{cat} Value');
title('Bootstrapped k_{cat} Values');
set(gca, 'XTick', [1, 2], 'XTickLabel', {'No Cre', 'Cre'});
hold off;

% Plot dot plot with error bars for K_M for both conditions
figure;
hold on;
% range
% errorbar([1, 2], [lower_percentile_K_M_noCre, lower_percentile_K_M_Cre], [upper_percentile_K_M_noCre, upper_percentile_K_M_Cre], 'bo', 'LineWidth', 1.5);
% stdev
errorbar([1, 2], [mean(bootstrap_params_K_M_noCre), mean(bootstrap_params_K_M_Cre)], [std_K_M_noCre, std_K_M_Cre], 'bo', 'LineWidth', 1.5);
xlabel('Condition');
ylabel('K_{M} Value');
title('Bootstrapped K_{M} Values');
set(gca, 'XTick', [1, 2], 'XTickLabel', {'No Cre', 'Cre'});
hold off;

fprintf('Mean bootstrapped parameters:\n');
fprintf('  k_cat (no Cre): %f\n', mean(bootstrap_params_k_cat_noCre));
fprintf('  k_cat (Cre): %f\n', mean(bootstrap_params_k_cat_Cre));
fprintf('  K_M (no Cre): %f\n', mean(bootstrap_params_K_M_noCre));
fprintf('  K_M (Cre): %d\n', mean(bootstrap_params_K_M_Cre));

%% t-test

% t-test on k_cat

% Perform a two-sample t-test between k_cat values for No Cre and Cre conditions
[h, p, ci, stats] = ttest2(bootstrap_params_k_cat_noCre, bootstrap_params_k_cat_Cre);

% Display the results
fprintf('T-test results for k_cat (No Cre vs Cre):\n');
fprintf('  p-value: %f\n', p);
fprintf('  95%% Confidence Interval for the difference in means: [%.4f, %.4f]\n', ci);
fprintf('  t-statistic: %f\n', stats.tstat);
fprintf('  Degrees of freedom: %d\n', stats.df);

% t-test on K_M

% Perform a two-sample t-test between K_M values for No Cre and Cre conditions
[h, p, ci, stats] = ttest2(bootstrap_params_K_M_noCre, bootstrap_params_K_M_Cre);

% Display the results
fprintf('T-test results for K_M (No Cre vs Cre):\n');
fprintf('  p-value: %f\n', p);
fprintf('  95%% Confidence Interval for the difference in means: [%.4f, %.4f]\n', ci);
fprintf('  t-statistic: %f\n', stats.tstat);
fprintf('  Degrees of freedom: %d\n', stats.df);



%% Functions

function predictions = f2(params, zf, alpha)

% unpack constants
k_cat_prime = params(1);
K_M = params(2);
alpha_prime = alpha;

% change to fix n
n=1;

predictions = k_cat_prime * zf.^n ./ (K_M + zf.^n) + alpha_prime;

end