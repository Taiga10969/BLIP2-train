class Config():
    def __init__(self) -> None:
        
        self.project_name = f"gpt2v_freez_deplot_normcap"
        self.batch_size = 18
        self.num_eposh = 10

        self.clip_value = 0.5

        self.max_length = 128

        self.lr = 5e-4  #学習率
        self.weight_decay = 0.1
        self.t0 = 20
