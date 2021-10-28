class aifi_style_interaction_detector:
    def __init__(self, confirmation_thresh=5) -> None:
        self.state = False # 0 no interaction 1 interaction
        self.confirmations = 0
        self.confirmation_thresh = confirmation_thresh
    
    def test(self, new_state):
        if new_state != self.state:
            self.confirmations += 1
            if self.confirmations > self.confirmation_thresh:
                self.confirmations = 0
                self.state = new_state
        else:
            self.confirmations = 0