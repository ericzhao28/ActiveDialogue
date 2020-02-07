"""Sampling strategies of thresholds on expected uncertainty reduction"""


def token_entropy_threshold(env, posterior, token_i):
  if posterior:
    current_dist = env.current_posterior_dist()[token_i]
  else:
    current_dist = env.current_dist()[token_i]

  labels_size = env.action_bounds[1]

  current_entropy = -np.dot(current_dist, np.log(current_dist + EPSILON))
  new_entropies = []
  for label in range(labels_size):
    # pdb.set_trace()
    # If positive feedback...
    mask = np.full((labels_size,), fill_value=0)
    for l in env.downstream_slots(label):
      mask[l] = 1
    new_dist = current_dist * mask
    new_dist = np.array(new_dist, dtype=np.float64) / sum(new_dist)
    if_true_entropy = -np.dot(new_dist, np.log(new_dist + EPSILON))

    # If negative feedback...
    mask = np.full((labels_size,), fill_value=1)
    for l in env.downstream_slots(label):
      mask[l] = 0
    new_dist = current_dist * mask
    new_dist = np.array(new_dist, dtype=np.float64) / sum(new_dist)
    if_false_entropy = -np.dot(new_dist, np.log(new_dist + EPSILON))

    # Compute expected entropy reduction
    odds_true = np.sum(
        current_dist[env.downstream_slots(label)])
    new_entropies.append(odds_true * if_true_entropy
                         + (1 - odds_true) * if_false_entropy)

  entropy_reductions = [
      current_entropy - new_entropy for new_entropy in new_entropies
  ]
  return np.max(entropy_reductions), np.argmax(entropy_reductions)


def entropy_threshold(env, threshold, posterior):
  best_i, best_entropy, best_j = None, None, None
  thresholds = [token_entropy_threshold(env, posterior, i)
                for i in range(env.seq_len())]
  for j, (e, i) in enumerate(thresholds):
    if best_i is None or e > best_entropy:
      best_i = i
      best_j = j
      best_entropy = e

  action = env.noop_label()
  if best_entropy > threshold:
    action[best_j] = best_i
  return [action]
