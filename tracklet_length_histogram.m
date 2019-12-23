clear, clc

generic_path = '/Users/bcs/Desktop/Repos/MSc/bcs_tracker/dukemtmc/purity_results%s/camera%d/tl%d_iou%0.5g_d%0.5g.txt';

cam_id = 1;
tracklet_lengths = [5, 10, 15, 20, 30, 40, 50];
ious = [0.95, 0.9, 0.75, 0.6, 0.5];
cos_dists = [0.5, 0.3, 0.1, 0.05, 0.005];





ious = [0.95, 0.75, 0.5];
cos_dists = [0.3, 0.1, 0.05];

cos_axis_length = size(cos_dists, 2);
iou_axis_length = size(ious, 2);
tracklet_lengths_length = size(tracklet_lengths, 2)



figure('NumberTitle', 'off', 'Name', sprintf('Camera %d', cam_id));

ci = 1

for ioui = 1:iou_axis_length
    for ti = 1:tracklet_lengths_length
        subplot_id = (ioui-1)*tracklet_lengths_length + ti;
        subplot(iou_axis_length, tracklet_lengths_length, subplot_id);
        purities = zeros(size(tracklet_lengths));
        
        
        clear A
        current_path = sprintf(generic_path, '', cam_id, tracklet_lengths(ti), ious(ioui), cos_dists(ci));
        A = load(current_path);
        
        histogram(A(:, 3), 'BinLimits', [1, tracklet_lengths(ti)])    
        
        
        grid on
        plot_title = sprintf('iou: %.6g, d: %.6g', ious(ioui), cos_dists(ci));
        title(plot_title)
        
        xlabel('Tracklet Length')
        ylabel('Count')
    end
end

sgtitle(sprintf('Camera %d', cam_id))

