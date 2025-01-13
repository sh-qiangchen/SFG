DONE_FILE="done_commands.txt"

if [ ! -f "$DONE_FILE" ]; then
    touch "$DONE_FILE"
fi

mapfile -t done_commands < "$DONE_FILE"

commands=(
	# german SAGE
	"python fairvgnn.py --dataset='german' --encoder='SAGE' --d_epochs=5 --g_epochs=10 --c_epochs=10 --c_lr=0.001 --e_lr=0.01 --ratio=0.5"
	"python fairvgnn.py --dataset='german' --encoder='SAGE' --d_epochs=5 --g_epochs=10 --c_epochs=10 --c_lr=0.001 --e_lr=0.01 --ratio=0.5 --with_constraint --rho=20 --loss_alpha=0.4"
    "python fairvgnn.py --dataset='german' --encoder='SAGE' --d_epochs=5 --g_epochs=10 --c_epochs=10 --c_lr=0.001 --e_lr=0.01 --ratio=0.5 --with_constraint --rho=20 --loss_alpha=0.5"
    "python fairvgnn.py --dataset='german' --encoder='SAGE' --d_epochs=5 --g_epochs=10 --c_epochs=10 --c_lr=0.001 --e_lr=0.01 --ratio=0.5 --with_constraint --rho=20 --loss_alpha=0.6"

    "python fairvgnn.py --dataset='german' --encoder='SAGE' --d_epochs=5 --g_epochs=10 --c_epochs=10 --c_lr=0.001 --e_lr=0.01 --ratio=0.5 --loss_alpha=0.5"
	
	# bail SAGE
	"python fairvgnn.py --dataset='bail' --encoder='SAGE' --d_epochs=5 --g_epochs=10 --c_epochs=5 --c_lr=0.01 --e_lr=0.001 --ratio=1 --epochs=160"
    "python fairvgnn.py --dataset='bail' --encoder='SAGE' --d_epochs=5 --g_epochs=10 --c_epochs=5 --c_lr=0.01 --e_lr=0.001 --ratio=1 --epochs=160 --with_constraint --rho=4 --loss_alpha=0.4"
    "python fairvgnn.py --dataset='bail' --encoder='SAGE' --d_epochs=5 --g_epochs=10 --c_epochs=5 --c_lr=0.01 --e_lr=0.001 --ratio=1 --epochs=160 --with_constraint --rho=4 --loss_alpha=0.5"   # best
    "python fairvgnn.py --dataset='bail' --encoder='SAGE' --d_epochs=5 --g_epochs=10 --c_epochs=5 --c_lr=0.01 --e_lr=0.001 --ratio=1 --epochs=160 --with_constraint --rho=4 --loss_alpha=0.6"
	    
	# credit SAGE
    "python fairvgnn_credit.py --dataset='credit' --encoder='SAGE' --d_epochs=5 --g_epochs=5 --c_epochs=10 --c_lr=0.01 --e_lr=0.001 --ratio=0"
	"python fairvgnn_credit.py --dataset='credit' --encoder='SAGE' --d_epochs=5 --g_epochs=5 --c_epochs=10 --c_lr=0.01 --e_lr=0.001 --ratio=0 --with_constraint --rho=4 --loss_alpha=0.4"
    "python fairvgnn_credit.py --dataset='credit' --encoder='SAGE' --d_epochs=5 --g_epochs=5 --c_epochs=10 --c_lr=0.01 --e_lr=0.001 --ratio=0 --with_constraint --rho=4 --loss_alpha=0.5"
    "python fairvgnn_credit.py --dataset='credit' --encoder='SAGE' --d_epochs=5 --g_epochs=5 --c_epochs=10 --c_lr=0.01 --e_lr=0.001 --ratio=0 --with_constraint --rho=4 --loss_alpha=0.6"
  )

for cmd in "${commands[@]}"; do
    if [[ ! " ${done_commands[*]} " =~ " ${cmd} " ]]; then
        echo "Executing: $cmd"
        eval $cmd
        echo "$cmd" >> "$DONE_FILE"
    else
        echo "Skipping: $cmd (already done)"
    fi
done

echo "All commands completed."