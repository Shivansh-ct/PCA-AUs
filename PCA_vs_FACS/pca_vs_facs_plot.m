dict_name = 'delta_pureAUs';
dict_name = 'delta_combAUs';
%dict_name = 'delta_pcaAUs';
%data_name = 'delta_CK+';
data_name = 'delta_BP4D_full';

larsen = @larsen;

T = readtable(strcat('Data/', dict_name, '.csv'));
D = table2array(T(2:end, 2:end));
% D = randn(136, 136);


T = readtable(strcat('Data/', data_name, '.csv'));
X = table2array(T(2:end, 2:end));

 
if strcmp(dict_name, 'delta_pureAUs') && strcmp(data_name, 'delta_CK+')
    cols_to_drop = [1, 17, 28, 40, 43, 37, 46];
    
elseif strcmp(dict_name, 'delta_pureAUs') && strcmp(data_name, 'delta_BP4D_full')
    cols_to_drop = [1:17, 28, 34, 40, 43, 37, 46, 61, 65]; 
    
elseif strcmp(dict_name, 'delta_combAUs') && strcmp(data_name, 'delta_CK+')
    cols_to_drop = [1, 17, 28, 40, 43, 37, 46];
    
elseif strcmp(dict_name, 'delta_combAUs') && strcmp(data_name, 'delta_BP4D_full')
    cols_to_drop = [1:17, 28, 34, 40, 43, 37, 46, 61, 65]; 
    
elseif strcmp(dict_name, 'delta_pcaAUs') && strcmp(data_name, 'delta_CK+')
    cols_to_drop = [1, 17, 28, 40, 43, 37, 46, 61, 65]; 
    
elseif strcmp(dict_name, 'delta_pcaAUs') && strcmp(data_name, 'delta_BP4D_full')
    cols_to_drop = [1:17, 28, 34, 40, 43, 37, 46, 61, 65];
    
else
    error('Invalid input strings'); % Throw an error for invalid input strings
end


% Add the 68 index columns to the columns to be dropped
cols_to_drop = [cols_to_drop, [cols_to_drop+68]];

% Remove the specified columns from X and D
X(:, cols_to_drop) = [];
D(:, cols_to_drop) = [];

D = transpose(D);

Norm = zeros(size(X,1), size(D,2));
Norm_X = 0;

for i=1:size(X,1)

    disp(strcat(dict_name,'-',data_name,'-',string(i)));

    temp_X = X(i, :);
    temp_X = reshape(temp_X, size(temp_X,2), 1);

    try
        [b steps] = larsen(D, temp_X, 0, -size(D,2), [], true, false);

        B = zeros(size(D,2), size(D,2));
        for k = 1:size(D,2)
        col = b(:, sum(b~=0, 1) == k);
        if ~isempty(col)
            B(:, k) = col(:, end);
        end
    end


    diff = D*B-temp_X;
    for j=1:size(D,2)
        Norm(i, j) = norm(diff(:,j),2);
    end

    Norm_X = Norm_X+norm(temp_X,2);

    catch e
        disp(e);
    end

end

final = zeros(size(D,2),1);

for i=1:size(D,2)
    final(i) = 100*(1-(sum(Norm(:,i),1)/Norm_X));
end

csvwrite(strcat('Results/',dict_name,'_',data_name, '_var.csv'), final);
