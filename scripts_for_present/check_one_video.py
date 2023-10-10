from show_results_model import get_frames_with_bb


if __name__ == '__main__':
    weights_path = '../scripts/weights_ver5.pt'
    get_frames_with_bb(video_path='../prepair_dataset/train/train_in_out\\24b468923c4ae72b.mp4',
                       path_to_dir='../problem_frames',
                       path_to_weights=weights_path)