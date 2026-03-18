% traffic.m
%
% Runs HOSVD compression on the traffic dataset for two permutations of the
% tensor modes, matching the experiment setup in experiments.sh. Results are
% appended to experiments.csv in the same format as the Python experiments.
%
% Dataset: data/traffic.bin
%   Binary file written by traffic_writer.m.
%   Header: uint32 height, uint32 width, float64 fps.
%   Body: float64 values in column-major order, shape (H, W, 3, T).
%
% Permutations (matching experiments.sh):
%   0123 -> (H, W, 3, T) = (120, 160, 3, 120) — original order
%   0321 -> (H, T, 3, W) = (120, 120, 3, 160) — axes 1 and 3 swapped
%
% For each permutation and each tol value, HOSVD is run via the MATLAB
% Tensor Toolbox (hosvd.m). Compression ratio and error metrics are
% computed and appended to experiments.csv.
%
% Requirements:
%   - MATLAB Tensor Toolbox (https://www.tensortoolbox.org) on the MATLAB path
%   - data/traffic.bin present relative to the scripts/ directory
%
% Usage:
%   cd scripts/
%   matlab -nodisplay -nosplash -r "traffic; exit"

% --- Read traffic.bin ---
fid = fopen('../data/traffic.bin', 'r');
height = fread(fid, 1, 'uint32');
width  = fread(fid, 1, 'uint32');
fps    = fread(fid, 1, 'double');  % read but not used; advances file pointer past header
data   = fread(fid, inf, 'double');
fclose(fid);

% Derive number of frames from total data size (not stored in header)
frameSize = height * width * 3;
numFrames = length(data) / frameSize;

% Reshape flat vector into 4D tensor (H x W x 3 x T)
% MATLAB reshape is column-major by default, matching how traffic_writer.m wrote the data
A = reshape(data, [height, width, 3, numFrames]);
fprintf('Tensor dimensions: %d x %d x %d x %d\n', size(A,1), size(A,2), size(A,3), size(A,4));

% tol values matching tsvdmii runs in experiments.sh
tol_values = [0.0001, 0.001, 0.01, 0.025, 0.050, 0.1, 0.25, 0.5];

% --- Open CSV for appending ---
csv_file = 'experiments.csv';
fcsv = fopen(csv_file, 'a');

% Write header only if file is empty
finfo = dir(csv_file);
if finfo.bytes == 0
    fprintf(fcsv, 'filename,complete,dname_f,alg_f,k_or_tol_f,mtype_f,perm_mode_f,omp_threads_f,alg,mtype,k,tol,dname,dfile,perm_mode,tensor_shape,time_transform_matrix,time_data_to_pystarm,compression_ratio,time_compress_ttm_total,time_slicewise_svd,time_slicewise_svdvals,time_thresholds,time_slicewise_svdks,time_reconstruct_matmul,time_reconstruct_ttm_total,absolute_err,relative_err,norm_original,norm_reconstructed\n');
end

% Two permutations matching experiments.sh traffic runs
% MATLAB permute uses 1-based indices, so (0,1,2,3) -> [1,2,3,4] and (0,3,2,1) -> [1,4,3,2]
perm_modes  = {[1,2,3,4], [1,4,3,2]};
perm_labels = {'0123', '0321'};

for p = 1:2
    % Apply permutation and convert to Tensor Toolbox tensor
    A_perm = permute(A, perm_modes{p});
    T = tensor(A_perm);
    original_size = numel(A_perm);
    norm_A = norm(T);  % Frobenius norm of original, used for relative error
    tensor_shape = sprintf('(%d, %d, %d, %d)', size(A_perm,1), size(A_perm,2), size(A_perm,3), size(A_perm,4));

    fprintf('\nperm_mode: %s — tensor shape: %s\n', perm_labels{p}, tensor_shape);
    fprintf('%-10s %-20s %-20s %-20s\n', 'tol', 'compression_ratio', 'absolute_err', 'relative_err');

    for i = 1:length(tol_values)
        tol = tol_values(i);

        % Run HOSVD with relative error-based truncation at the given tolerance
        T_approx = hosvd(T, tol);

        % Compressed size = core tensor + all factor matrices
        compressed_size = numel(T_approx.core);
        for m = 1:length(T_approx.U)
            compressed_size = compressed_size + numel(T_approx.U{m});
        end

        compression_ratio = original_size / compressed_size;

        % Reconstruct full tensor and compute error metrics
        diff = T - full(T_approx);
        abs_err = norm(diff);           % ||A - A_approx||_F
        rel_err = abs_err / norm_A;     % ||A - A_approx||_F / ||A||_F
        norm_reconst = norm(full(T_approx));

        fprintf('%-10g %-20.6f %-20.6f %-20.6f\n', tol, compression_ratio, abs_err, rel_err);

        % Append row to CSV
        % _f columns (1-8) are left empty as this is a MATLAB experiment (no log filename to parse)
        % CSV columns (30 total):
        % 1:filename, 2:complete, 3:dname_f, 4:alg_f, 5:k_or_tol_f, 6:mtype_f,
        % 7:perm_mode_f, 8:omp_threads_f, 9:alg, 10:mtype, 11:k, 12:tol,
        % 13:dname, 14:dfile, 15:perm_mode, 16:tensor_shape,
        % 17:time_transform_matrix, 18:time_data_to_pystarm, 19:compression_ratio,
        % 20-26:timings(empty), 27:absolute_err, 28:relative_err,
        % 29:norm_original, 30:norm_reconstructed
        fprintf(fcsv, ',,,,,,,,hosvd,,,%g,traffic,../data/traffic.bin,%s,"%s",,,%g,,,,,,,,%g,%g,%g,%g\n', ...
            tol, perm_labels{p}, tensor_shape, compression_ratio, abs_err, rel_err, norm_A, norm_reconst);
    end
end

fclose(fcsv);
