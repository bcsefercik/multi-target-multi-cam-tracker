clear, clc

% generic_path = '/Users/bcs/Desktop/Repos/MSc/bcs_tracker/dukemtmc/purity_results%s/camera%d/tl%d_iou%0.5g_d%0.5g.txt';
generic_path = '/Users/bcs/Desktop/Repos/MSc/purity_results/openpose_60fps_val%s/camera%d/tl%d_iou%0.5g_d%0.5g.txt';

% ious = [0.95, 0.9, 0.75, 0.6, 0.5];
% cos_dists = [0.5, 0.3, 0.1, 0.05, 0.005];

% tracklet_lengths = [5, 10, 15, 20, 30, 40, 50];
tracklet_lengths = [5 10 15 20 30 60 120];
ious = [0.75, 0.85];
cos_dists = [0.4];

cos_axis_length = size(cos_dists, 2);
iou_axis_length = size(ious, 2);
tracklet_lengths_length = size(tracklet_lengths, 2);


for camid = 1:8
% for camid = [1 2 3 5 6 7 8]
for ci = 1:cos_axis_length
figure('NumberTitle', 'off', 'Name', sprintf('Camera: %d, Distance: %.4g', camid, cos_dists(ci)));

for ioui = 1:iou_axis_length
    for ti = 1:tracklet_lengths_length
        subplot_id = (ioui-1)*tracklet_lengths_length + ti;
        subplot(iou_axis_length, tracklet_lengths_length, subplot_id);
        purities = zeros(size(tracklet_lengths));
        
        
        clear A
        current_path = sprintf(generic_path, '', camid, tracklet_lengths(ti), ious(ioui), cos_dists(ci));
        A = load(current_path);
        
        histogram(A(:, 3), 'BinLimits', [1, min(1000, tracklet_lengths(ti))]) 
%         histogram(A(:, 3))    
        
        
        grid on
        plot_title = sprintf('iou: %.6g, d: %.6g', ious(ioui), cos_dists(ci));
        title(plot_title)
        
        xlabel_str = sprintf('Tracklet Length\nfull/total: %.6g', sum(max(A(:, 3))==A(:, 3))/size(A(:, 3),1));
        xlabel(xlabel_str)
        ylabel('Count')
    end
end

end
end
