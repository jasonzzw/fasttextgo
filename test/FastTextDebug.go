package main

import (
	"fmt"

	"git.garena.com/common/gocommon/goutil"
	"github.com/jasonzzw/fasttextgo"
)

func main() {
	fasttextgo.LoadModel("ft_tw_model.bin")
	result, err := fasttextgo.Predict("mildlin 12色", 10)
	if err == nil {
		sorted := goutil.SortedKeysDescFloat32(result)
		for _, cluster := range sorted {
			fmt.Printf("%s=%.2f\n", cluster, result[cluster])
		}
	} else {
		fmt.Printf("prediction err=%s\n", err)
	}
}
