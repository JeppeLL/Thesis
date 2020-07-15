# Convert uploaded images to TFRecords
import dataset_tool
import pretrained_networks
import dnnlib
import dnnlib.tflib as tflib
dataset_tool.create_from_images("/zhome/ca/6/92701/Desktop/Master_Thesis/stylegan2/rl_images/256/image_for_latents/records/", "/zhome/ca/6/92701/Desktop/Master_Thesis/stylegan2/rl_images/256/image_for_latents/images/", True)

# Run the projector
import run_projector
import projector
import training.dataset
import training.misc
import os 

_G, _D, Gs = pretrained_networks.load_networks("/zhome/ca/6/92701/Desktop/Master_Thesis/stylegan2/Old_SG2_pkl/network-snapshot-012820.pkl")


def project_real_images(dataset_name, data_dir, num_images, num_snapshots):
    proj = projector.Projector()
    proj.set_network(Gs)

    print('Loading images from "%s"...' % dataset_name)
    dataset_obj = training.dataset.load_dataset(data_dir=data_dir, tfrecord_dir=dataset_name, max_label_size=0, verbose=True, repeat=False, shuffle_mb=0)
    assert dataset_obj.shape == Gs.output_shape[1:]

    for image_idx in range(num_images):
        print('Projecting image %d/%d ...' % (image_idx, num_images))
        images, _labels = dataset_obj.get_minibatch_np(1)
        images = training.misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
        run_projector.project_image(proj, targets=images, png_prefix=dnnlib.make_run_dir_path('Old_SG2_pkl/projection/image%04d-' % image_idx), num_snapshots=num_snapshots)

project_real_images("records","/zhome/ca/6/92701/Desktop/Master_Thesis/stylegan2/rl_images/256/image_for_latents/",1,10)

