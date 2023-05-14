from torchgems.mp_pipeline import train_model
import torch


class train_model_master:
    def __init__(
        self,
        model_gen1,
        model_gen2,
        local_rank,
        batch_size,
        epochs,
        criterion=None,
        optimizer=None,
        parts=1,
        ASYNC=True,
        replications=1,
    ):
        self.mp_size = model_gen1.split_size
        self.split_size = model_gen1.split_size
        self.second_rank = self.split_size - local_rank - 1

        self.train_model1 = train_model(
            model_gen1,
            local_rank,
            batch_size,
            epochs,
            criterion=None,
            optimizer=None,
            parts=parts,
            ASYNC=True,
            GEMS_INVERSE=False,
        )
        self.train_model2 = train_model(
            model_gen2,
            self.second_rank,
            batch_size,
            epochs,
            criterion=None,
            optimizer=None,
            parts=parts,
            ASYNC=True,
            GEMS_INVERSE=True,
        )

        # self.train_model1.models = self.train_model1.models.to('cpu')

        # self.train_model2.models = self.train_model2.models.to('cpu')

        self.parts = parts
        self.epochs = epochs
        self.local_rank = local_rank
        self.ENABLE_ASYNC = ASYNC
        self.batch_size = batch_size

        self.replications = replications

        # self.initialize_recv_buffers()
        # self.initialize_send_recv_ranks()

    def run_step(self, inputs, labels):
        loss, correct = 0, 0
        # torch.cuda.empty_cache()

        # self.train_model1.models = self.train_model1.models.to('cuda')
        temp_loss, temp_correct = self.train_model1.run_step(
            inputs[: self.batch_size], labels[: self.batch_size]
        )
        loss += temp_loss
        correct += temp_correct

        # torch.cuda.empty_cache()

        # self.train_model1.models = self.train_model1.models.to('cpu')
        # self.train_model2.models = self.train_model2.models.to('cuda')
        temp_loss, temp_correct = self.train_model2.run_step(
            inputs[self.batch_size : 2 * self.batch_size],
            labels[self.batch_size : 2 * self.batch_size],
        )

        # self.train_model2.models = self.train_model2.models.to('cpu')

        # torch.cuda.empty_cache()

        loss += temp_loss
        correct += temp_correct

        torch.cuda.synchronize()
        for times in range(self.replications - 1):
            index = (2 * times) + 2
            temp_loss, temp_correct = self.train_model1.run_step(
                inputs[index * self.batch_size : (index + 1) * self.batch_size],
                labels[index * self.batch_size : (index + 1) * self.batch_size],
            )
            loss += temp_loss
            correct += temp_correct

            temp_loss, temp_correct = self.train_model2.run_step(
                inputs[(index + 1) * self.batch_size : (index + 2) * self.batch_size],
                labels[(index + 1) * self.batch_size : (index + 2) * self.batch_size],
            )

            loss += temp_loss
            correct += temp_correct
        return loss, correct
