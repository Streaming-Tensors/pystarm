% Write the ..../toolbox/images/imdata/traffic.mj2 file in binary format
videoFile = 'traffic.mj2';
v = VideoReader(videoFile);

height = v.Height;
width = v.Width;
fps = v.FrameRate;

fid = fopen('traffic.bin', 'w');

% Write header to keep H, W, and fps
fwrite(fid, height, 'uint32');
fwrite(fid, width,  'uint32');
fwrite(fid, fps,    'double');

% Write each frame (storing in double)
% NOTE: each pixel value is uint8, so we are casting to double
while hasFrame(v)
    frame = readFrame(v);
    fwrite(fid, frame, 'double');
end

fclose(fid);