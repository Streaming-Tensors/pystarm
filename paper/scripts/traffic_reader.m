fid = fopen('traffic.bin', 'r');

% Read metadata
height = fread(fid, 1, 'uint32');
width  = fread(fid, 1, 'uint32');
fps    = fread(fid, 1, 'double');

% Read rest as double
data = fread(fid, inf, 'double');
fclose(fid);

% Reshape
frameSize = height * width * 3;
numFrames = length(data) / frameSize;

% H x W x 3 (RGB channels) x Frames
videoData = reshape(data, [height, width, 3, numFrames]);