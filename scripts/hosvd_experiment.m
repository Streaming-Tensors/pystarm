% hosvd_experiment.m
%
% Runs a single HOSVD experiment for one (tol, perm_str, dname_str) combination.
% Prints results to stdout in the same format as experiments.py so that
% parse_logs.py can parse the log file without any changes.
%
% Called by experiments.sh once per parameter combination. The caller sets
% tol, perm_str, dname_str, and dfile_str in the workspace via the -r flag:
%
%   matlab -nodisplay -nosplash -r \
%     "addpath('...'); tol=0.0001; perm_str='0123'; dname_str='traffic-color'; \
%      dfile_str='/path/to/traffic.bin'; run('scripts/hosvd_experiment.m'); exit"
%
% Inputs (set in workspace by caller):
%   alg_str   - algorithm: 'hosvd' or 'hosvd-proportional'
%   tol       - relative energy tolerance for 'hosvd' (e.g. 0.0001)
%   p         - proportional rank factor for 'hosvd-proportional' (e.g. 0.125);
%               ranks are set to ceil(dim_i * p) for each mode i
%   perm_str  - permutation mode string
%   dname_str - dataset name: 'traffic-color', 'traffic-gray', or 'dcmall'
%   dfile_str - full path to the data file
%
% Requirements:
%   - MATLAB Tensor Toolbox on the MATLAB path (addpath before calling)

% --- Locate data file relative to this script ---
% --- Read data based on dname_str ---
if strcmp(dname_str, 'traffic-color') || strcmp(dname_str, 'traffic-gray')
    % Read traffic.bin (custom binary format written by traffic_writer.m)
    fid    = fopen(dfile_str, 'r');
    height = fread(fid, 1, 'uint32');
    width  = fread(fid, 1, 'uint32');
    fps    = fread(fid, 1, 'double');  % read but not used; advances past header
    data   = fread(fid, inf, 'double');
    fclose(fid);

    frameSize = height * width * 3;
    numFrames = length(data) / frameSize;
    color = reshape(data, [height, width, 3, numFrames]);

    if strcmp(dname_str, 'traffic-color')
        % 4-way color tensor (H x W x 3 x T), normalized
        A = color / norm(tensor(color));

        switch perm_str
            case '0123', perm = [1,2,3,4]; perm_label = '(0, 1, 2, 3)';
            case '0321', perm = [1,4,3,2]; perm_label = '(0, 3, 2, 1)';
            otherwise,   error('Unsupported perm_str for traffic-color: %s', perm_str);
        end

    else  % traffic-gray
        % Convert to grayscale using BT.601 luminance weights.
        % Note: im2gray is not used here because traffic.bin stores pixel values as
        % float64 in [0,255] (cast from uint8 by traffic_writer.m). im2gray expects
        % double input in [0,1], so passing raw values would be incorrect without
        % first casting to uint8 or dividing by 255. Applying the BT.601 coefficients
        % directly avoids the cast and produces identical results.
        gray = 0.298936021293775 * color(:,:,1,:) + ...
               0.587043074451121 * color(:,:,2,:) + ...
               0.114020904255103 * color(:,:,3,:);
        % squeeze: (H x W x 1 x T) -> (H x W x T)
        A = squeeze(gray);
        A = A / norm(tensor(A));

        switch perm_str
            case '012', perm = [1,2,3]; perm_label = '(0, 1, 2)';
            case '021', perm = [1,3,2]; perm_label = '(0, 2, 1)';
            case '120', perm = [2,3,1]; perm_label = '(1, 2, 0)';
            otherwise,  error('Unsupported perm_str for traffic-gray: %s', perm_str);
        end
    end

elseif strcmp(dname_str, 'dcmall')
    % Read hyperspectral TIF file; imread returns (bands x height x width)
    A = double(imread(dfile_str));

    switch perm_str
        case '021', perm = [1,3,2]; perm_label = '(0, 2, 1)';
        otherwise,  error('Unsupported perm_str for dcmall: %s', perm_str);
    end

else
    error('Unsupported dname_str: %s', dname_str);
end

% --- Apply permutation ---
A_perm        = permute(A, perm);
original_size = numel(A_perm);

sz = size(A_perm);
tensor_shape_parts = arrayfun(@(x) num2str(x), sz, 'UniformOutput', false);
tensor_shape = ['(' strjoin(tensor_shape_parts, ', ') ')'];

% --- Print parameters (matching experiments.py format for parse_logs.py) ---
fprintf('alg         : %s\n', alg_str);
fprintf('mtype        : eye\n');
fprintf('k            : None\n');
if strcmp(alg_str, 'hosvd')
    fprintf('tol          : %g\n', tol);
else
    fprintf('tol          : %g\n', p);
end
fprintf('dname        : %s\n', dname_str);
fprintf('dfile        : %s\n', dfile_str);
fprintf('perm_mode    : %s\n', perm_label);
fprintf('Tensor shape: %s\n', tensor_shape);

% --- Run HOSVD ---
T      = tensor(A_perm);
norm_A = norm(T);

if strcmp(alg_str, 'hosvd')
    T_approx = hosvd(T, tol);
elseif strcmp(alg_str, 'hosvd-proportional')
    ranks = arrayfun(@(d) ceil(d * p), sz);
    T_approx = hosvd(T, 0, 'ranks', ranks);
else
    error('Unsupported alg_str: %s', alg_str);
end

% --- Compression ratio: original elements / compressed elements ---
compressed_size = numel(T_approx.core);
for m = 1:length(T_approx.U)
    compressed_size = compressed_size + numel(T_approx.U{m});
end
compression_ratio = original_size / compressed_size;

% --- Error metrics ---
T_full      = full(T_approx);
diff_tensor = T - T_full;
abs_err     = norm(diff_tensor);
rel_err     = abs_err / norm_A;
norm_reconst = norm(T_full);

% --- Print results (matching experiments.py format for parse_logs.py) ---
fprintf('Compression ratio: %.15g\n', compression_ratio);
fprintf('Absolute err: %.15g\n', abs_err);
fprintf('Relative err: %.15g\n', rel_err);
fprintf('Norm of original tensor: %.15g\n', norm_A);
fprintf('Norm of reconstructed tensor: %.15g\n', norm_reconst);
