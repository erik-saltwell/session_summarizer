
# add setting for not only attendees but who they play


# Improve registration
 - stich together short clips to set some kind of length something
 - remove outlier clips
 - stitch into one giant file
 - Only take clips with a high enough similarity residual

 # Commands
 - create-speaker-clips<change: only high residual clips>
   - speaker-clip-minimum-similarity-residual : float (default:0.2) - the minimum cosine similarity residual before a clip is used as a speaker clip
 - merge-speaker-clips: This command takes a speaker label and an output folder. It will repeatedly merge the shortest clips for a speaker until all clips for that speaker are longer then the minimum-speaker-clip-duration setting value
    - minimum-speaker-clip-duration: float (default 2.0) - when merging short clips, the system will merge the shortest clips until no clips are shorter than this value
 - remove-outlier-speaker-clips: This command takes a speaker label and an output foler. It will repeatedly compute the centroid embedding across all clips and then copy over all clips except those that are outliers.
    - stable-centroid-epsilon: float (default: 0.001) The threshold difference between centroid similarity cosines that determines if removed clips 'changed' the centroid
 - register-speaker-clips <change>


# confirm we only use attendees when we identify speakers


