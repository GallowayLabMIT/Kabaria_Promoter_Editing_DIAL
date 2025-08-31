%% Modeling for ZF37 titration

%% all summary plot, ZF37

% for only summary plot data
% reporter - 1
% ZF37 - 2
% bioreplicate - 3
% mGL-A_gmean - 4
% mCherry-A_gmean - 5
% Fraction - 6
% Count - 7
% Normalized_mGL-A_gmean - 8
% Normalized_mCherry-A_gmean - 9

% code starts here

% % biorep0
% filepath_noCre = '/Users/yunbeenbae/Documents/GitHub/Cre-loxP-ZF-promoters/flow_analysis_scripts/2024_promoter_editing_paper/datafiles/modeling_noCre_ZF37tit_biorep0.xlsx';
% filepath_Cre = '/Users/yunbeenbae/Documents/GitHub/Cre-loxP-ZF-promoters/flow_analysis_scripts/2024_promoter_editing_paper/datafiles/modeling_Cre_ZF37tit_biorep0.xlsx';
% 
% % biorep1
% filepath_noCre = '/Users/yunbeenbae/Documents/GitHub/Cre-loxP-ZF-promoters/flow_analysis_scripts/2024_promoter_editing_paper/datafiles/modeling_noCre_ZF37tit_biorep1.xlsx';
% filepath_Cre = '/Users/yunbeenbae/Documents/GitHub/Cre-loxP-ZF-promoters/flow_analysis_scripts/2024_promoter_editing_paper/datafiles/modeling_Cre_ZF37tit_biorep1.xlsx';

% % biorep2
% filepath_noCre = '/Users/yunbeenbae/Documents/GitHub/Cre-loxP-ZF-promoters/flow_analysis_scripts/2024_promoter_editing_paper/datafiles/modeling_noCre_ZF37tit_biorep2.xlsx';
% filepath_Cre = '/Users/yunbeenbae/Documents/GitHub/Cre-loxP-ZF-promoters/flow_analysis_scripts/2024_promoter_editing_paper/datafiles/modeling_Cre_ZF37tit_biorep2.xlsx';

% % biorep3
% filepath_noCre = '/Users/yunbeenbae/Documents/GitHub/Cre-loxP-ZF-promoters/flow_analysis_scripts/2024_promoter_editing_paper/datafiles/modeling_noCre_ZF37tit_biorep3.xlsx';
% filepath_Cre = '/Users/yunbeenbae/Documents/GitHub/Cre-loxP-ZF-promoters/flow_analysis_scripts/2024_promoter_editing_paper/datafiles/modeling_Cre_ZF37tit_biorep3.xlsx';

% biorep4
filepath_noCre = '/Users/yunbeenbae/Documents/GitHub/Cre-loxP-ZF-promoters/flow_analysis_scripts/2024_promoter_editing_paper/datafiles/modeling_noCre_ZF37tit_biorep4.xlsx';
filepath_Cre = '/Users/yunbeenbae/Documents/GitHub/Cre-loxP-ZF-promoters/flow_analysis_scripts/2024_promoter_editing_paper/datafiles/modeling_Cre_ZF37tit_biorep4.xlsx';

data_noCre = readmatrix(filepath_noCre);
data_Cre = readmatrix(filepath_Cre);

x_zf_noCre = data_noCre(:,9); % mCherry-A
y_rna_noCre = data_noCre(:,8); % mGL-A

x_zf_Cre = data_Cre(:,9); % mCherry-A
y_rna_Cre = data_Cre(:,8); % mGL-A

%% removing burden points

% gate visually
zf_burden = 7; % selected upon visual inspection of graph
positions = find(x_zf_Cre > zf_burden);

% remove the points related to burden
x_zf_Cre(positions) = [];
y_rna_Cre(positions) = [];

%% initializing mCherry expression

x_zf_Cre = x_zf_Cre-x_zf_Cre(1);
x_zf_noCre = x_zf_noCre-x_zf_noCre(1);

%% setting basal expression

alpha_noCre = y_rna_noCre(1);
alpha_Cre = y_rna_Cre(1);

%% fit the curve to noCre points

objective_function = @(params, zf) f2(params, zf, alpha_noCre);

% k_cat_prime, K_M
params_guess = [1, 0.05];

params_noCre = lsqcurvefit(objective_function, params_guess, x_zf_noCre, y_rna_noCre);

% calculate estimated reporter expression based on fitted parameters
predictions_noCre = f2(params_noCre, x_zf_noCre, alpha_noCre);

% repeat for Cre points
objective_function = @(params, zf) f2(params, zf, alpha_Cre);

% k_cat_prime, K_M
params_guess = [14, 0.2];
params_Cre = lsqcurvefit(objective_function, params_guess, x_zf_Cre, y_rna_Cre);

predictions_Cre = f2(params_Cre, x_zf_Cre, alpha_Cre);

%% Plot the data and model predictions

plot(x_zf_noCre, y_rna_noCre, 'ro', 'DisplayName', 'No Cre Data', 'MarkerSize',3);
hold on;
plot(x_zf_noCre, predictions_noCre, 'r-', 'DisplayName', 'No Cre Model Predictions');
xlabel('mCherry-A Normalized by gmean of ZF37 0.125x');
ylabel('mGL-A  Normalized by gmean of ZF37 0.125x');
title('Scatterplot Normalized by ZF37 0.125x with Fitted model');
legend;
set(gca, 'XScale', 'linear'); % Set x-axis to logarithmic scale
set(gca, 'YScale', 'linear'); % Set y-axis to logarithmic scale
grid on;
hold on;
% Plot the data and model predictions
plot(x_zf_Cre, y_rna_Cre, 'bo', 'DisplayName', 'Cre Data', 'MarkerSize',3);
hold on;
plot(x_zf_Cre, predictions_Cre, 'g-', 'DisplayName', 'Cre Model Predictions');
hold off;

%% Display parameters

labels = {'k_cat_prime','K_M'};

disp('No Cre parameters')
disp('Index   Value   Label');
for i = 1:length(params_noCre)
    fprintf('%d\t\t%d\t\t%s\n', i, params_noCre(i), labels{i});
end

disp('Cre parameters')
disp('Index   Value   Label');
for i = 1:length(params_Cre)
    fprintf('%d\t\t%d\t\t%s\n', i, params_Cre(i), labels{i});
end

disp('alpha')
fprintf('no Cre %d\n', alpha_noCre)
fprintf('Cre %d\n',alpha_Cre)

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